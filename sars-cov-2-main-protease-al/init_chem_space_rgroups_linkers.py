import fegrow
from rdkit import Chem
from multiprocessing import Pool
import itertools as it




def load_molecules():
    # Load R-group data
    ld_rgroups = [fegrow.RGroupGrid._load_molecules().Mol.values, fegrow.RGroupGrid._load_molecules().Rgroup_id.values]
    rgroups, rgroup_idx = ld_rgroups

    # Load linker data
    ld_linkers = [fegrow.RLinkerGrid._load_molecules().Mol.values, fegrow.RLinkerGrid._load_molecules().Common.values]
    linkers, linkers_idx = ld_linkers
    linkers = linkers[:40]
    linkers_idx = linkers_idx[:40]

    return rgroups, rgroup_idx, linkers, linkers_idx


def build_smiles(args):
    h, rgroup_id, linker_id = args
    try:
        rgroup = rgroups[rgroup_id]
        linker = linkers[linker_id]
        core_linker = fegrow.build_molecules(scaffold, linker, [h])[0]
        new_mol = fegrow.build_molecules(core_linker, rgroup)[0]
        smiles = Chem.MolToSmiles(new_mol)
        return smiles, h, rgroup_id, linker_id
    except Exception as e:
        print(f"Error in building SMILES: {e}")
        return None, h, rgroup_id, linker_id


if __name__ == '__main__':
    rgroups, rgroup_idx, linkers, linkers_idx = load_molecules()
    scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
    hs = [6]  # Hydrogen indices

    all_combos = it.product(hs, range(len(rgroup_idx)), range(len(linkers_idx)))

    with Pool(20) as p:
        results = p.map(build_smiles, all_combos)

    with open(f'id_h{hs[0]}_rgroups_linkers.csv', 'w') as OUT:
        OUT.write('Smiles,h,RGroupIndex,LinkerIndex\n')
        for result in results:
            if result[0]:  # Check if SMILES was successfully generated
                OUT.write(f'{result[0]},{result[1]},{result[2]},{result[3]}\n')

'''
ld_rgroups = list([fegrow.RGroupGrid._load_molecules().Mol.values, fegrow.RGroupGrid._load_molecules().Rgroup_id.values])
rgroups = ld_rgroups[0]
rgroup_idx = ld_rgroups[1]

ld_linkers = list([fegrow.RLinkerGrid._load_molecules().Mol.values, fegrow.RLinkerGrid._load_molecules().Common.values])
linkers = ld_linkers[0]
linkers_idx = ld_linkers[1]

prune_linkers = 40
linkers = linkers[:prune_linkers]

scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
hs = [6] #[a.GetIdx() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 1]


def build_smiles(args):
    h, rgroup_idx, linker_idx = args
    rgroup = rgroups[rgroup_idx]
    linker = linkers[linker_idx]
    core_linker = fegrow.build_molecules(scaffold, linker, [h])[0]
    new_mol = fegrow.build_molecules(core_linker, rgroup)[0]
    smiles = Chem.MolToSmiles(new_mol)
    return smiles, h, rgroup_idx, linker_idx

if __name__ == '__main__':
    for h in hs:
        all_combos = it.product([h], range(len(rgroups)), range(len(linkers)))
        with Pool(20) as p:
            results = p.map(build_smiles, all_combos)

        with open(f'id_h{h}_rgroups_linkers.csv', 'w') as OUT:
            OUT.write('Smiles,h,RGroupIndex,LinkerIndex\n')
            for smiles, h, rgroup_idx, linker_idx in results:
                OUT.write(f'{smiles},{h},{rgroup_idx},{linker_idx}\n')
'''