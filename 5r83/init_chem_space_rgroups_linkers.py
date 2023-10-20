import fegrow
import time
from rdkit import Chem
import pandas as pd
from multiprocessing import Pool
from functools import partial
import itertools as it
from rdkit.Chem.FilterCatalog import SmartsMatcher

rgroups = list(fegrow.RGroupGrid._load_molecules().Mol.values)
linkers = list(fegrow.RLinkerGrid._load_molecules().Mol.values)

linkers = linkers[:400]
h = 6 
scaffold = Chem.SDMolSupplier('5r83_coreh.sdf', removeHs=False)[0]
oo_matcher = SmartsMatcher('Oxygen-Oxygen', '[#8]-[#8]', minCount=0, maxCount=0)
ss_matcher = SmartsMatcher('Sulphur-Sulphur', '[#16]-[#16]', minCount=0, maxCount=0)
nitron_matcher = SmartsMatcher('NO2+-N', '[#7+]-[#7]', minCount=0, maxCount=0)

def build_smiles(args):
    h, rgroup, linker = args
    core_linker = fegrow.build_molecules(scaffold, linker, [h])[0]
    new_mol = fegrow.build_molecules(core_linker, rgroup)[0]
    if not oo_matcher.HasMatch(new_mol) or not ss_matcher.HasMatch(new_mol) or not nitron_matcher.HasMatch(new_mol):
        smiles, h = None, None
    else:
        smiles = Chem.MolToSmiles(new_mol)
    return smiles, h


if __name__ == '__main__':
    all_combos = it.product([h], rgroups, linkers)
    with Pool(28) as p:
        results = p.map(build_smiles, all_combos)

    #if smiles or h == None, remove 
    filtered_results = [(smiles, h) for smiles, h in results if smiles is not None and h is not None]


    with open('manual_init.csv', 'w') as OUT:
        OUT.write('Smiles,h\n')
        for smiles, h in filtered_results:
            OUT.write(f'{smiles},{h}\n')
