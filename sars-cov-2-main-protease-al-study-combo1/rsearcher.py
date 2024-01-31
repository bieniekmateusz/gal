"""
FG_SCALE - if fewer tasks are in the queue than this, generate more. This also sets that many "proceses" in Dask.

 - Paths:
export FG_GNINA_PATH=/path/to/gnina
"""
import copy
import time
import sys
import tempfile
import os
from pathlib import Path
import logging
import datetime
import threading
import queue
import cProfile
import functools

import dask
from dask import array
import numpy
from dask.distributed import Client, performance_report
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import ml_collections
import parmed
import warnings
from plip.structure.preparation import PDBComplex

import fegrow

import al_for_fep.models.sklearn_gaussian_process_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from mycluster import create_cluster
except ImportError:
    def create_cluster():
        from dask.distributed import LocalCluster
        return LocalCluster()


# xstal interactions from 24 mpro fragments
mpro_crystal_structures_interactions = {'hacceptor_THR_25_8',
                                         'hdonor_THR_25_8',
                                         'hydrophobic_THR_25',
                                         'hacceptor_HIS_41_8',
                                         'hydrophobic_HIS_41',
                                         'pistacking_HIS_41',
                                         'hacceptor_CYS_44_8',
                                         'hydrophobic_PHE_140',
                                         'hacceptor_ASN_142_8',
                                         'hdonor_ASN_142_7',
                                         'hdonor_GLY_143_7',
                                        'hacceptor_SER_144_8',
                                        'hdonor_SER_144_7',
                                        'hdonor_SER_144_8',
                                        'hdonor_CYS_145_7',
                                        'hacceptor_HIS_163_7',
                                        'hydrophobic_MET_165',
                                        'hacceptor_GLU_166_8',
                                        'hdonor_GLU_166_7',
                                        'hdonor_GLU_166_8',
                                        'hydrophobic_GLU_166',
                                        'saltbridge_GLU_166',
                                        'hydrophobic_PRO_168',
                                        'hydrophobic_ASP_187',
                                        'hacceptor_GLN_189_8',
                                        'hdonor_GLN_189_7',
                                        'hydrophobic_GLN_189'}


def plip_mpro_merge_score(protein, ligand):
    with tempfile.TemporaryDirectory() as TD:

        # turn the rdkit.Mol into an SDF file
        if isinstance(ligand, Chem.Mol):
            ligand_path = os.path.join(TD, "ligand.sdf")
            with Chem.SDWriter(ligand_path) as SD:
                SD.write(ligand)
                ligand = ligand_path

        lig = parmed.load_file(ligand)
        if isinstance(lig, list):
            warnings.warn("The ligand was an array (SDF?). Using the first frame. ")
            lig = lig[0]
        protein = parmed.load_file(protein)
        system = protein + lig
        complex_path = os.path.join(TD, "complex.pdb")
        system.save(complex_path, renumber=False)
        return plip_mpro_score(str(complex_path))

def plip_mpro_score(complex_path):
    """
    Get all the protein interactions from the interactions between the protein and the ligand

    Args:
        complex_path:

    Returns: A list of strings
    """
    complex = PDBComplex()
    complex.load_pdb(complex_path) # Load the PDB file into PLIP class
    complex.analyze()

    # assume there is only one ligand for now
    if len(complex.interaction_sets) != 1:
        raise ValueError("PLIP detected more (or less) than one ligand?!")

    # pair key and values
    interactions = list(complex.interaction_sets.values())[0]

    # take all the interactions
    hydrophobic_contacts = ["hydrophobic_" + c.restype + "_" + str(c.resnr) for c in interactions.hydrophobic_contacts]
    # extract protein donors
    hdonors = ["hdonor_" + d.restype + "_" + str(d.resnr) + "_" + str(d.d.atomicnum) for d in interactions.hbonds_pdon]
    # extract protein acceptors
    hacceptors = ["hacceptor_" + a.restype + "_" + str(a.resnr) + "_" + str(a.a.atomicnum) for a in interactions.hbonds_ldon]
    pistacking = ["pistacking_" + r.restype + "_" + str(r.resnr) for r in interactions.pistacking]
    saltbridge = ["saltbridge_" + r.restype + "_" + str(r.resnr) for r in  interactions.saltbridge_pneg]
    waterbridge = ["waterbridge_" + r.restype + "_" + str(r.resnr) for r in interactions.water_bridges]
    pication = ["pication_" + r.restype + "_" + str(r.resnr) for r in interactions.pication_paro]
    halogen_bond = ["halogenbond_" + r.restype + "_" + str(r.resnr) for r in interactions.halogen_bonds]
    metal_complex = ["metalcomplex_" + r.restype + "_" + str(r.resnr) for r in interactions.metal_complexes]

    protein_interaction_fingerprints = (hydrophobic_contacts +
                                        hdonors +
                                        hacceptors +
                                        pistacking +
                                        saltbridge +
                                        waterbridge +
                                        pication +
                                        halogen_bond +
                                        metal_complex)

    intersection = len(mpro_crystal_structures_interactions.intersection(protein_interaction_fingerprints))
    count = len(mpro_crystal_structures_interactions) + len(protein_interaction_fingerprints)
    tanimoto_distance = intersection / (count - intersection)

    return protein_interaction_fingerprints, tanimoto_distance

@dask.delayed
def evaluate(scaffold, h, smiles, pdb_filename, gnina_path):
    t_start = time.time()
    print(f'TIME changed dir: {time.time() - t_start:.1f}s')
    fegrow.RMol.set_gnina(gnina_path)

    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    rmol = fegrow.RMol(Chem.MolFromSmiles(smiles, params=params))

    # remove the h
    scaffold = copy.deepcopy(scaffold)
    scaffold_m = Chem.EditableMol(scaffold)
    scaffold_m.RemoveAtom(int(h))
    scaffold = scaffold_m.GetMol()
    rmol._save_template(scaffold)

    rmol.generate_conformers(num_conf=50, minimum_conf_rms=0.5)
    rmol.remove_clashing_confs(pdb_filename)

    rmol.optimise_in_receptor(
        receptor_file=pdb_filename,
        ligand_force_field="openff",
        use_ani=True,
        sigma_scale_factor=0.8,
        relative_permittivity=4,
        water_model=None,
        platform_name='CPU',
    )

    if rmol.GetNumConformers() == 0:
        raise Exception("No Conformers")

    rmol.sort_conformers(energy_range=2) # kcal/mol
    gnina_affinity = rmol.gnina(receptor_file=pdb_filename)
    ifp, ifp_score = plip_mpro_merge_score(pdb_filename, rmol)

    MW = Descriptors.MolWt(rmol)

    data = {
        # the active learner minimises the values
        "cnnaffinity": gnina_affinity.CNNaffinity[0],
        "plip": -ifp_score,
        "filename": Chem.MolToSmiles(rmol, allHsExplicit=True),
        # (CNN/MW) * (PLIP/0.3) * 100
        # note that we make it negative, the lower the better,
        "combo1": -(gnina_affinity.CNNaffinity[0]/MW) * (max(ifp_score, 0.000001)/0.3) * 100
    }

    print(f'TIME Completed the molecule generation in {time.time() - t_start:.1f}s.')
    return rmol, data


def get_saving_queue():
    # start a saving queue (just for Rocket, which struggles with saving file speed)
    mol_saving_queue = queue.Queue()

    def worker_saver():
        while True:
            mol = mol_saving_queue.get()
            filename = mol.GetProp("filename")
            with Chem.SDWriter(f'structures/{filename}.sdf') as SD:
                SD.write(mol)
                print(f'Wrote {filename}')

            mol_saving_queue.task_done()

    threading.Thread(target=worker_saver, daemon=False).start()
    return mol_saving_queue

@dask.delayed
def compute_fps(fingerprint_radius, fingerprint_size, smiless):
    fingerprint_fn = functools.partial(
        AllChem.GetMorganFingerprintAsBitVect,
        radius=fingerprint_radius,
        nBits=fingerprint_size)
    return numpy.array([fingerprint_fn(Chem.MolFromSmiles(smiles)) for smiles in smiless])


def dask_parse_feature_smiles_morgan_fingerprint(feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
    smiless = feature_dataframe[feature_column].values
    print(f"About to compute fingerprints for {len(smiless)} smiles ")
    start = time.time()

    workers_num = max(sum(client.nthreads().values()), 30) # assume minimum 30 workers
    results = client.compute([compute_fps(fingerprint_radius, fingerprint_size, smiles)
                              for smiles in
                              numpy.array_split(    # split smiless between the workers
                                  smiless, min(workers_num, len(smiless)))])
    fingerprints = numpy.concatenate([result.result() for result in results])
    print(f"Computed {len(fingerprints)} fingerprints in {time.time() - start}")
    return fingerprints


def dask_tanimito_similarity(a, b):
    print(f"About to compute tanimoto for array lengths {len(a)} and {len(b)}")
    start = time.time()
    chunk_size = 8_000
    da = array.from_array(a, chunks=chunk_size)
    db = array.from_array(b, chunks=chunk_size)
    aa = array.sum(da, axis=1, keepdims=True)
    bb = array.sum(db, axis=1, keepdims=True)
    ab = array.matmul(da, db.T)
    td = array.true_divide(ab, aa + bb.T - ab)
    td_computed = td.compute()
    print(f"Computed tanimoto similarity in {time.time() - start:.2f}s for array lengths {len(a)} and {len(b)}")
    return td_computed

def get_config():
  return ml_collections.ConfigDict({
      'model_config':
          ml_collections.ConfigDict({
              'model_type': 'gp',
              'hyperparameters': ml_collections.ConfigDict(),
              'tuning_hyperparameters': ml_collections.ConfigDict(),
              'features':
                  ml_collections.ConfigDict({
                      'feature_type': 'fingerprint',
                      'params': {
                          'feature_column': 'Smiles',
                          'fingerprint_size': 2048,
                          'fingerprint_radius': 3
                      }
                  }),
              'targets':
                  ml_collections.ConfigDict({
                      'feature_type': 'number',
                      'params': {
                          'feature_column': 'combo1',
                      }
                  })
          }),
      'selection_config':
          ml_collections.ConfigDict({
              'selection_type': 'UCB',      # greedy / uncertainty based
              'hyperparameters': ml_collections.ConfigDict({"beta": 10}), # 0 ~= greedy, 0.1 = exploit,  10 = explore
              'num_elements': 200,						# n mols per cycle
              'selection_columns': ["combo1", "Smiles", 'h', 'enamine_id', 'enamine_searched']
          }),
      'metadata': 'Small test for active learning.',
      'cycle_dir': '',
      'training_pool': '',
      'virtual_library': '',
      'diverse': True,				# initial diverse set
  })

if __name__ == '__main__':
    # overwrite internals with Dask methods
    import al_for_fep.data.utils
    al_for_fep.data.utils.parse_feature_smiles_morgan_fingerprint = dask_parse_feature_smiles_morgan_fingerprint
    al_for_fep.models.sklearn_gaussian_process_model._tanimoto_similarity = dask_tanimito_similarity

    import mal
    config = get_config()
    config.virtual_library = "manual_init_h6_rgroups_linkers500.csv"    #   initial_chemical_space
    feature = 'combo1'
    config.model_config.targets.params.feature_column = feature

    pdb_filename = dask.delayed(str(Path('rec_final.pdb').absolute()))
    scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
    scaffold_dask = dask.delayed(scaffold)
    gnina_path = dask.delayed(os.environ['FG_GNINA_PATH'])

    t_now = datetime.datetime.now()
    client = Client(create_cluster())
    print('Client', client)

    al = mal.ActiveLearner(config, client)
    jobs = {}
    next_selection = None
    mol_saving_queue = get_saving_queue()
    while True:
        print(f"{datetime.datetime.now() - t_now}: Queue: {len(jobs)} tasks ")

        for job, args in list(jobs.items()):
            if not job.done():
                continue

            # get back the original arguments
            smiles = jobs[job]
            del jobs[job]

            try:
                rmol, rmol_data = job.result()
                [rmol.SetProp(k, str(v)) for k, v in rmol_data.items()]
                mol_saving_queue.put(rmol)
                score = float(rmol_data[feature])
                print("Success: molecule evaluated. ")
            except Exception as E:
                log.warning("Failed to evaluate the molecule. Assigning the penalty 0. "
                            "Error: " + str(E))
                score = 0   # penalty

            al.virtual_library.loc[al.virtual_library.Smiles == smiles, [feature, 'Training']] = score, True

        if len(jobs) == 0:
            print(f'Iteration finished. Next.')

            # save the results from the previous iteration
            if next_selection is not None:
                for i, row in next_selection.iterrows():
                    # bookkeeping
                    next_selection.loc[next_selection.Smiles == row.Smiles, [feature, 'Training']] = \
                        al.virtual_library[al.virtual_library.Smiles == row.Smiles][feature].values[0], True
                al.csv_cycle_summary(next_selection)

            # for the best scoring molecules, add molecules from Enamine that are similar
            # this way we ensure that the Enamine molecules will be evaluated
            al.add_enamine_molecules(scaffold=scaffold)

            # cProfile.run('next_selection = al.get_next_best()', filename='get_next_best.prof', sort=True)
            next_selection = al.get_next_best()

            # select 20 random molecules
            for_submission = [evaluate(scaffold_dask, row.h, row.Smiles, pdb_filename, gnina_path) for _, row in next_selection.iterrows()]
            for job, (_, row) in zip(client.compute(for_submission), next_selection.iterrows()):
                jobs[job] = row.Smiles

        time.sleep(5)
