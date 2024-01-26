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
import openmm.app
import pandas as pd
import ml_collections

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

@dask.delayed
def evaluate(scaffold, h, smiles, pdb_filename, gnina_path):
    t_start = time.time()
    print(f'TIME changed dir: {time.time() - t_start:.1f}s')
    fegrow.RMol.set_gnina(gnina_path)

    params = Chem.SmilesParserParams()
    params.removeHs = False  # keep the hydrogens
    rmol = fegrow.RMol(Chem.MolFromSmiles(smiles, params=params))

    if "FG_DEBUG" in os.environ:
        return rmol, {"cnnaffinity": -10}

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
    affinities = rmol.gnina(receptor_file=pdb_filename)
    data = {
        "cnnaffinity": -max(affinities.CNNaffinity),
        "filename": Chem.MolToSmiles(rmol),
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
                          'feature_column': 'cnnaffinity',
                      }
                  })
          }),
      'selection_config':
          ml_collections.ConfigDict({
              'selection_type': 'UCB',      # greedy / uncertainty based
              'hyperparameters': ml_collections.ConfigDict({"beta": 0.1}), # 0 ~= greedy, 0.1 = exploit,  10 = explore
              'num_elements': 200,						# n mols per cycle
              'selection_columns': ["cnnaffinity", "Smiles", 'h', 'enamine_id', 'enamine_searched']
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
    feature = 'cnnaffinity'
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
