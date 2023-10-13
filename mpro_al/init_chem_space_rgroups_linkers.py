import fegrow
import time
from rdkit import Chem
import pandas as pd
from multiprocessing import Pool
from functools import partial
import itertools as it

rgroups = list(fegrow.RGroupGrid._load_molecules().Mol.values)
linkers = list(fegrow.RLinkerGrid._load_molecules().Mol.values)

linkers = linkers[:10]

scaffold = Chem.SDMolSupplier('coreh.sdf', removeHs=False)[0]

def build_smiles(args):
    h, rgroup, linker = args
    core_linker = fegrow.build_molecules(scaffold, linker, [h])[0]
    new_mol = fegrow.build_molecules(core_linker, rgroup)[0]
    smiles = Chem.MolToSmiles(new_mol)
    return smiles, h


if __name__ == '__main__':
    all_combos = it.product([6], rgroups, linkers)
    with Pool(28) as p:
        results = p.map(build_smiles, all_combos)

    with open('manual_init.csv', 'w') as OUT:
        OUT.write('Smiles,h\n')
        for smiles, h in results:
            OUT.write(f'{smiles},{h}\n')