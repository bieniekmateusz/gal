"""
FG_SCALE - if fewer tasks are in the queue than this, generate more. This also sets that many "proceses" in Dask.

 - Paths:
export FG_ANIMODEL_PATH=/path/to/animodel.pt
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
import openmm.app
import pandas as pd

import fegrow

import al_for_fep.models.sklearn_gaussian_process_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    # get hardware specific cluster
    from mycluster import create_cluster
except ImportError:
    # set up a local cluster just in case
    def create_cluster():
        from dask.distributed import LocalCluster
        return LocalCluster()
        # lc = LocalCluster(processes=True, threads_per_worker=1, n_workers=2)
        # lc.adapt(maximum_cores=2)
        # return lc

# Define the modified sf1 function to operate on a dictionary
def sf1(rmol_data_dict):
    # Assert rmol_data_dict contains the necessary data
    required_keys = ['plip', 'interactions', 'cnn_ic50_norm', 'MW', 'QED', 'cnnaffinity']
    missing_keys = [key for key in required_keys if key not in rmol_data_dict]

    if missing_keys:
        raise ValueError(f"Missing necessary data: {', '.join(missing_keys)}")

    # Calculate the score
    sf1 = -rmol_data_dict['cnnaffinity'] / rmol_data_dict['MW'] #rmol_data_dict['plip'] * rmol_data_dict['QED'] * rmol_data_dict['cnnaffinity']#* rmol_data_dict['plip'] * (1 + (0.5 * len(rmol_data_dict['interactions']))) * (
          #      10 ** (rmol_data_dict['cnnaffinity'] / rmol_data_dict['MW']))

    return sf1

# xstal interactions from 24 mpro fragments
xstal_interaction_list = ['hbond_ASN_142_O2',
 'hbond_ASN_142_O3',
 'hbond_CYS_145_O3',
 'hbond_CYS_44_O2',
 'hbond_GLN_189_O2',
 'hbond_GLN_189_O3',
 'hbond_GLU_166_N3',
 'hbond_GLU_166_O.co2',
 'hbond_GLU_166_O2',
 'hbond_GLU_166_O3',
 'hbond_GLY_143_O3',
 'hbond_HIS_163_N2',
 'hbond_HIS_41_O2',
 'hbond_SER_144_O3',
 'hbond_THR_25_N3',
 'hbond_THR_25_O3',
 'hydrophobic_ASP_187_nan',
 'hydrophobic_GLN_189_nan',
 'hydrophobic_GLU_166_nan',
 'hydrophobic_HIS_41_nan',
 'hydrophobic_MET_165_nan',
 'hydrophobic_PHE_140_nan',
 'hydrophobic_PRO_168_nan',
 'hydrophobic_THR_25_nan',
 'pistacking_HIS_41_nan',
 'saltbridge_GLU_166_nan']
xstal_set = set(xstal_interaction_list)

def plip_score(ref_set, var_set):
    """
    Compute Tanimoto similarity between a reference set of xstal plip interactions & another molecule.

    Parameters:
    ref_set (set): Reference set
    var_set (set): Variable set to compare

    Returns:
    float: Tanimoto similarity score
    """
    # Convert sets to single-column DataFrames
    ref_df = pd.DataFrame({0: list(ref_set)})
    var_df = pd.DataFrame({0: list(var_set)})

    # Ensure DataFrames have only one column
    assert len(ref_df.columns) == 1 and len(var_df.columns) == 1, "DataFrames should have only one column"

    # Compute intersection
    common_elements = len(ref_set.intersection(var_set))

    # Compute the total number of elements in the reference set
    total_elements = len(ref_set)

    if total_elements == 0:
        return 0  # Edge case: both sets are empty

    # Calculate Tanimoto similarity
    tanimoto_similarity = common_elements / total_elements

    return tanimoto_similarity





def gen_intrns_dict(interactions):
    """
    Generate an interaction dictionary sorted by residue number and secondarily by residue name.

    Parameters:
    interactions (list): List of interaction strings

    Returns:
    dict: Sorted interaction dictionary mapping interaction strings to bit positions
    """
    # extract RESNR and RESNAME from interactions using regex
    interactions['interaction'] = interactions.apply(
        lambda row: f"{row['type']}_{row['RESTYPE']}_{str(row['RESNR'])}_{row['ACCEPTORTYPE']}", axis=1)
    interaction_tuples = []
    print(interaction_tuples)
    for interaction in interactions['interaction']:
        match = re.search('_(\w+)_(\d+)', interaction)
        if match:
            interaction_tuples.append((match.groups(), interaction))
        else:
            print(f"Skipping malformed interaction: {interaction}")

    # sort by RESNR  and RESNAME
    sorted_interactions = sorted(interaction_tuples, key=lambda x: (int(x[0][1]), x[0][0]))

    # dictionary mapping sorted interactions to bit positions
    interaction_dict = {interaction: index for index, (_, interaction) in enumerate(sorted_interactions)}

    return interaction_dict



@dask.delayed
def evaluate(scaffold, h, smiles, pdb_load):
    t_start = time.time()
    fegrow.RMol.set_gnina(os.environ['FG_GNINA_PATH'])
    with tempfile.TemporaryDirectory() as TMP:
        TMP = Path(TMP)
        os.chdir(TMP)
        print(f'TIME changed dir: {time.time() - t_start:.1f}s')

        # make a symbolic link to animodel
        # ani = Path(os.environ['FG_ANIMODEL_PATH'])
        # Path('animodel.pt').symlink_to(ani)
        # print(f'TIME linked animodel: {time.time() - t_start}')

        protein = str(TMP / 'protein_tmp.pdb')
        with open(protein, 'w') as PDB:
            PDB.write(pdb_load)

        params = Chem.SmilesParserParams()
        params.removeHs = False  # keep the hydrogens
        rmol = fegrow.RMol(Chem.MolFromSmiles(smiles, params=params))
        # remove the h
        scaffold = copy.deepcopy(scaffold)
        scaffold_m = Chem.EditableMol(scaffold)
        scaffold_m.RemoveAtom(int(h))
        scaffold = scaffold_m.GetMol()
        rmol._save_template(scaffold)
        print(f'TIME prepped rmol: {time.time() - t_start:.1f}s')

        rmol.generate_conformers(num_conf=50, minimum_conf_rms=0.2)
        rmol.remove_clashing_confs(protein)
        print(f'TIME conformers done: {time.time() - t_start:.1f}s')

        rmol.optimise_in_receptor(
            receptor_file=protein,
            ligand_force_field="openff",
            use_ani=True,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name='CPU',
        )

        if rmol.GetNumConformers() == 0:
            raise Exception("No Conformers")

        print(f'TIME opt done: {time.time() - t_start:.1f}s')
        rmol.sort_conformers(energy_range=2) # kcal/mol
        affinities = rmol.gnina(receptor_file=protein)
        data = {
            "cnnaffinity": affinities.CNNaffinity.values[0],
            "cnnaffinityIC50": affinities["CNNaffinity->IC50s"].values[0],
            "hydrogens": [atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomicNum() == 1],
            "interactions": None,
            "plip": None,
            "cnn_ic50_norm": None,
            "MW": None,
            "cnnic50": None,
            "sf1": None,

        }
        # other props
        tox = rmol.toxicity()
        tox = dict(zip(list(tox.columns), list(tox.values[0])))
        tox['MW'] = Descriptors.HeavyAtomMolWt(rmol)
        for k, v in tox.items():
            data[k] = v
            # compute all props
            print(f'Calculating PLIP')

            plip_itrns = rmol.plip_interactions(receptor_file=protein)
            plip_dict = gen_intrns_dict(plip_itrns)
            data['interactions'] = set(plip_dict.keys())
            data['plip'] = 0
            data['plip'] += plip_score(xstal_set, data['interactions']) * 50  # HYPERPARAM TO CHANGE
            data['cnn_ic50_norm'] = data['cnnaffinityIC50'] / data['MW']
            data['sf1'] = sf1(data)
            return rmol, data
        print(f'TIME Completed the molecule generation in {time.time() - t_start:.1f}s.')
        return rmol, data


def get_saving_queue():
    # start a saving queue (just for Rocket, which struggles with saving file speed)
    mol_saving_queue = queue.Queue()

    def worker_saver():
        while True:
            mol = mol_saving_queue.get()
            filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

            if os.path.exists(f'structures/{filename}.sdf'):
                print('Overwriting the existing file')

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


if __name__ == '__main__':
    # overwrite internals with Dask methods
    import al_for_fep.data.utils
    al_for_fep.data.utils.parse_feature_smiles_morgan_fingerprint = dask_parse_feature_smiles_morgan_fingerprint
    al_for_fep.models.sklearn_gaussian_process_model._tanimoto_similarity = dask_tanimito_similarity

    import mal
    from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
    config = get_gaussian_process_config()
    initial_chemical_space = "id_h6_rgroups_linkers.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 3  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h', 'enamine_id', 'enamine_searched']
    feature = 'cnnaffinity'
    config.model_config.targets.params.feature_column = feature
    config.model_config.features.params.fingerprint_size = 2048

    pdb_load = dask.delayed(open('rec_final.pdb').read())
    scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
    scaffold_dask = dask.delayed(scaffold)

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
                print('ERROR when scoring. Assigning a penalty. Error: ', E)
                score = 0 # penalty

            # print(f"Updating: {al.virtual_library.loc[al.virtual_library.Smiles == smiles]}, {smiles}")
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
            next_selection = al.get_next_best(diverse_start=True)

            # select 20 random molecules
            for_submission = [evaluate(scaffold_dask, row.h, row.Smiles, pdb_load) for _, row in next_selection.iterrows()]
            for job, (_, row) in zip(client.compute(for_submission), next_selection.iterrows()):
                jobs[job] = row.Smiles

        time.sleep(5)

    mol_saving_queue.join()
