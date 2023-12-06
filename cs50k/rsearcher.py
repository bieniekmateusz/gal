"""
Using a known chemical space, carry out hyperparameter scanning for AL.
"""
import time
import functools

import dask
from dask import array as darray
from dask.distributed import Client
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


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
    return np.array([fingerprint_fn(Chem.MolFromSmiles(smiles)) for smiles in smiless])


def dask_parse_feature_smiles_morgan_fingerprint(feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
    smiless = feature_dataframe[feature_column].values
    print(f"About to compute fingerprints for {len(smiless)} smiles ")
    start = time.time()

    # determine the number of connected workers
    workers_num = client.nthreads().values()

    # if it is adaptive, set the number of workers to 30
    if client.cluster._adaptive is not None:
        workers_num = max(workers_num, 30)

    results = client.compute([compute_fps(fingerprint_radius, fingerprint_size, smiles)
                              for smiles in
                              np.array_split(    # split smiless between the workers
                                  smiless, min(workers_num, len(smiless)))])
    fingerprints = np.concatenate([result.result() for result in results])
    print(f"Computed {len(fingerprints)} fingerprints in {time.time() - start}")
    return fingerprints


def dask_tanimito_similarity(a, b):
    print(f"About to compute tanimoto for array lengths {len(a)} and {len(b)}")
    start = time.time()
    chunk_size = 8_000
    da = darray.from_array(a, chunks=chunk_size)
    db = darray.from_array(b, chunks=chunk_size)
    aa = darray.sum(da, axis=1, keepdims=True)
    bb = darray.sum(db, axis=1, keepdims=True)
    ab = darray.matmul(da, db.T)
    td = darray.true_divide(ab, aa + bb.T - ab)
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
    initial_chemical_space = "manual_init_h6_rgroups_linkers100_scorable.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 20  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h', 'fid']
    feature = 'cnnaffinity'
    config.model_config.targets.params.feature_column = feature
    config.model_config.features.params.fingerprint_size = 2048

    client = Client(create_cluster())
    print('Client', client)

    # load the precomputed chemical space
    oracle = pd.read_csv("cs_scored.csv", index_col="fid", usecols=["fid", "cnnaffinity"])

    al = mal.ActiveLearner(config, client)
    for iteration in range(3):
        print(f'> Iteration {iteration} finished. Next.')
        selection = al.get_next_best()

        # look up the precomputed values
        # use our FEgrow ID "fid" to identify each row
        selection.set_index("fid")
        selection['Training'] = True
        results = selection.merge(oracle, on=["fid"])
        # get the affinities, note that we make them negative for AL
        selection.cnnaffinity = -results.cnnaffinity_y.values
        np.testing.assert_array_less(selection.cnnaffinity, 0)

        # update the main library
        al.virtual_library.update(selection)
        al.virtual_library = al.virtual_library.astype({'Training': bool})

        al.csv_cycle_summary(selection)
