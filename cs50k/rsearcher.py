"""
Using a known chemical space, carry out hyperparameter scanning for AL.
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

try:
    # get hardware specific cluster
    from mycluster import create_cluster
except ImportError:
    # set up a local cluster just in case
    def create_cluster():
        from dask.distributed import LocalCluster
        return LocalCluster()


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
    initial_chemical_space = "manual_init_h6_rgroups_linkers100.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 20  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h', 'enamine_id', 'enamine_searched']
    feature = 'cnnaffinity'
    config.model_config.targets.params.feature_column = feature
    config.model_config.features.params.fingerprint_size = 2048

    client = Client(create_cluster())
    print('Client', client)

    al = mal.ActiveLearner(config, client)
    jobs = {}
    next_selection = None
    while True:
        if len(jobs) == 0:
            print(f'Iteration finished. Next.')

            # save the results from the previous iteration
            if next_selection is not None:
                for i, row in next_selection.iterrows():
                    # bookkeeping
                    next_selection.loc[next_selection.Smiles == row.Smiles, [feature, 'Training']] = \
                        al.virtual_library[al.virtual_library.Smiles == row.Smiles][feature].values[0], True
                al.csv_cycle_summary(next_selection)

            next_selection = al.get_next_best()

            # fixme - look up the values
            for_submission = [evaluate(next_selection) for _, row in next_selection.iterrows()]
            al.virtual_library.loc[al.virtual_library.Smiles == smiles, [feature, 'Training']] = score, True
        time.sleep(5)

    mol_saving_queue.join()
