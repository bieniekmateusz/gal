"""
FG_SCALE - if fewer tasks are in the queue than this, generate more. This also sets that many "proceses" in Dask.

 - Paths:
export FG_ANIMODEL_PATH=/path/to/animodel.pt
export FG_GNINA_PATH=/path/to/gnina
"""
import copy
import time
import random
import glob
import sys
import tempfile
import os
from pathlib import Path
import logging
import datetime
import threading
import queue

import dask
from dask.distributed import Client, performance_report
from rdkit import Chem
from rdkit.Chem import Descriptors
import openmm.app
import pandas as pd

import fegrow
import helpers
print('hi')
# get hardware specific cluster
try:
    from mycluster import create_cluster
except ImportError:
    # set up a local cluster just in case
    def create_cluster():
        from dask.distributed import LocalCluster
        lc = LocalCluster(processes=True, threads_per_worker=1)
        lc.adapt(maximum_cores=2)
        return lc

# preload the dataframes
rgroups = list(fegrow.RGroupGrid._load_molecules().Mol.values)
linkers = list(fegrow.RLinkerGrid._load_molecules().Mol.values)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NoConformers(Exception):
    pass


@dask.delayed
def score(scaffold, h, smiles, pdb_load):
    t_start = time.time()
    fegrow.RMol.set_gnina(os.environ['FG_GNINA_PATH'])
    with tempfile.TemporaryDirectory() as TMP:
        TMP = Path(TMP)
        os.chdir(TMP)
        print(f'TIME changed dir: {time.time() - t_start}')

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
        print(f'TIME prepped rmol: {time.time() - t_start}')

        rmol_data = helpers.Data()

        rmol.generate_conformers(num_conf=200, minimum_conf_rms=0.4)
        print('Number of simple conformers: ', rmol.GetNumConformers())

        rmol.remove_clashing_confs(protein)
        print(f'TIME conformers done: {time.time() - t_start}')
        print('Number of conformers after removing clashes: ', rmol.GetNumConformers())

        rmol.optimise_in_receptor(
            receptor_file=protein,
            ligand_force_field="openff",
            use_ani=False,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name='CPU',
        )

        # continue only if there are any conformers to be optimised
        if rmol.GetNumConformers() == 0:
            rmol_data.cnnaffinity = 10
            return rmol, rmol_data
        print(f'TIME opt done: {time.time() - t_start}')

        rmol.sort_conformers(energy_range=2) # kcal/mol
        affinities = rmol.gnina(receptor_file=protein)
        rmol_data.cnnaffinity = -affinities.CNNaffinity.values[0]
        rmol_data.cnnaffinityIC50 = affinities["CNNaffinity->IC50s"].values[0]
        rmol_data.hydrogens = [atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomicNum() == 1]

        # compute all props
        tox = rmol.toxicity()
        tox = dict(zip(list(tox.columns), list(tox.values[0])))
        tox['MW'] = Descriptors.HeavyAtomMolWt(rmol)
        for k, v in tox.items():
            setattr(rmol_data, k, v)
        rmol.interactions = rmol.plip_interactions(receptor_file="rec_final.pdb") # TODO write out pdb of receptor & use that
        # TODO make sure fegrow plip branch is merged or this wont wory

        print(f'Task: Completed the molecule generation in {time.time() - t_start} seconds. ')
        return rmol, rmol_data


def expand_chemical_space(al):
    """
    Expand the chemical space. This is another selection problem.
    Select the best performing "small", as the starting point ...
    """
    if al.virtual_library[al.virtual_library.Training == True].empty:
        return

    params = Chem.SmilesParserParams()
    params.removeHs = False
    for i, row in al.virtual_library[al.virtual_library.Training == True].iterrows():
        mol = Chem.MolFromSmiles(row.Smiles, params=params)
        Chem.AllChem.EmbedMolecule(mol)
        hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        smiles = build_smiles(mol, hs, rgroups)
        new_smiles = smiles.assign(Training=False)  # fixme: this will break if we use a different keyword

        extended = pd.concat([al.virtual_library, new_smiles], ignore_index=True)
        al.virtual_library = extended


def build_smiles(core, hs, groups):
    """
    Build a list of smiles that can be generated in theory
    """
    start = time.time()
    smiless = []
    hhooks = []
    for h in hs:
        for group in groups:
            for linker in linkers:
                core_linker = fegrow.build_molecules(core, linker, [h])[0]
                new_mol = fegrow.build_molecules(core_linker, group)[0]
                smiles = Chem.MolToSmiles(new_mol)
                smiless.append(smiles)
                hhooks.append(h)
    print('Generated initial smiles in: ', time.time() - start)
    return pd.DataFrame({'Smiles': smiless, 'h': hhooks})


def get_saving_queue():
    # start a saving queue (just for Rocket, which struggles with saving file speed)
    mol_saving_queue = queue.Queue()

    def worker_saver():
        while True:
            mol = mol_saving_queue.get()
            start = time.time()
            helpers.save(mol)
            print(f'Saved molecule in {time.time() - start}')
            mol_saving_queue.task_done()

    threading.Thread(target=worker_saver, daemon=False).start()
    return mol_saving_queue


if __name__ == '__main__':
    import mal
    from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
    config = get_gaussian_process_config()
    initial_chemical_space = "linker_500rgroup_4h.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 100  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h']
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    t_now = datetime.datetime.now()
    client = Client(create_cluster())
    print('Client', client)

    mol_saving_queue = get_saving_queue()

    # load the initial molecule
    scaffold = Chem.SDMolSupplier('structures/c1cn[nH]c1.sdf', removeHs=False)[0]
    scaffold_data = helpers.data(scaffold)

    if not os.path.exists(initial_chemical_space):
        smiles = build_smiles(scaffold, scaffold_data.hydrogens, rgroups)
        smiles.to_csv(initial_chemical_space, index=False)

    al = mal.ActiveLearner(config)

    futures = {}

    pdb_load = open('orighit_protein.pdb').read()

    next_selection = None
    while True:
        for future, args in list(futures.items()):
            if not future.done():
                continue

            # get back the original arguments
            scaffold, h, smiles, _ = futures[future]
            del futures[future]

            try:
                rmol, rmol_data = future.result()

                helpers.set_properties(rmol, rmol_data)
                mol_saving_queue.put(rmol)
                al.virtual_library.loc[al.virtual_library.Smiles == Chem.MolToSmiles(rmol),
                                       ['cnnaffinity', 'Training']] = float(rmol_data.cnnaffinity), True
            except Exception as E:
                print('ERROR: Will be ignored. Continuing the main loop. Error: ', E)
                continue

        print(f"{datetime.datetime.now() - t_now}: Queue: {len(futures)} tasks. ")

        if len(futures) == 0:
            print(f'Iteration finished. Next.')

            # expand_chemical_space(al)

            # save the results from the previous iteration
            if next_selection is not None:
                for i, row in next_selection.iterrows():
                    # bookkeeping
                    next_selection.loc[next_selection.Smiles == row.Smiles, ['cnnaffinity', 'Training']] = \
                        al.virtual_library[al.virtual_library.Smiles == row.Smiles].cnnaffinity.values[0], True
                al.csv_cycle_summary(next_selection)

            next_selection = al.get_next_best()

            # select 20 random molecules
            for i, row in next_selection.iterrows():
                args = [scaffold, row.h, row.Smiles, pdb_load]
                futures[client.compute([score(*args), ])[0]] = args

        time.sleep(5)

    mol_saving_queue.join()
