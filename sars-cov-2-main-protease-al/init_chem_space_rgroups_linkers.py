import fegrow
from rdkit import Chem
from multiprocessing import Pool
import itertools as it

rgroups = list(fegrow.RGroupGrid._load_molecules().Mol.values)
linkers = list(fegrow.RLinkerGrid._load_molecules().Mol.values)

prune_linkers = 500
linkers = linkers[:prune_linkers]

scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
hs = [a.GetIdx() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 1]

def build_smiles(args):
    h, rgroup, linker = args
    core_linker = fegrow.build_molecules(scaffold, linker, [h])[0]
    new_mol = fegrow.build_molecules(core_linker, rgroup)[0]
    smiles = Chem.MolToSmiles(new_mol)
    return smiles, h


if __name__ == '__main__':
    for h in hs:
        all_combos = it.product([h], rgroups, linkers)
        with Pool(20) as p:
            results = p.map(build_smiles, all_combos)

        with open(f'manual_init_h{h}_rgroups_linkers{prune_linkers}.csv', 'w') as OUT:
            OUT.write('Smiles,h\n')
            for smiles, h in results:
                OUT.write(f'{smiles},{h}\n')