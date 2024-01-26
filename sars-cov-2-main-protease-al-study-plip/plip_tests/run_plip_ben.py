import plip
import parmed
import tempfile
import os
from rdkit import Chem
from pathlib import Path
import pandas as pd
# mol = Chem.SDMolSupplier("structures/CC(C)(C)OC(=O)NC(=O)Sc1cccnc1.sdf")
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
import re

def create_df_from_binding_site(selected_site_interactions, interaction_type="hbond"):
    """
    Creates a data frame from a binding site and interaction type.

    Parameters
    ----------
    selected_site_interactions : dict
        Precaluclated interactions from PLIP for the selected site
    interaction_type : str
        The interaction type of interest (default set to hydrogen bond).

    Returns
    -------
    pd.DataFrame :
        DataFrame with information retrieved from PLIP.
    """

    # check if interaction type is valid:
    valid_types = [
        "hydrophobic",
        "hbond",
        "waterbridge",
        "saltbridge",
        "pistacking",
        "pication",
        "halogen",
        "metal",
    ]

    if interaction_type not in valid_types:
        raise ValueError("!!! Wrong interaction type specified. Hbond is chosen by default!!!\n")

    df = pd.DataFrame.from_records(
        # data is stored AFTER the column names
        selected_site_interactions[interaction_type][1:],
        # column names are always the first element
        columns=selected_site_interactions[interaction_type][0],
    )

    return df

def residue_interactions(pdb_id):
    interactions_by_site = retrieve_plip_interactions(f"{pdb_id}")

    index_of_selected_site = 0
    selected_site = list(interactions_by_site.keys())[index_of_selected_site]
    # print(selected_site)
    dfs = []
    valid_types = [
        "hydrophobic",
        "hbond",
        "waterbridge",
        "saltbridge",
        "pistacking",
        "pication",
        "halogen",
        "metal",
    ]
    for valid_type in valid_types:
        dfs.append(create_df_from_binding_site(interactions_by_site[selected_site], interaction_type=valid_type))

    all_interactions = pd.concat(dfs, keys=valid_types)

    # all_interactions = all_interactions.dropna(axis=1) #dont drop nan if want to keep hydrophobic interactions

    all_interactions = all_interactions.reset_index()
    all_interactions = all_interactions.rename(columns={"level_0": "type", "level_1": f"id"})
    # print(all_interactions)
    # oh = pd.get_dummies(a.index)  #onehot encode at some point potentially??

    #    res_interactions = all_interactions.groupby('RESNR')[['type','RESTYPE', 'DIST', 'ANGLE',]].agg(lambda x: list(x)).reset_index()
    #    res_interactions['id'] = pdb_id[x]

    return all_interactions

def retrieve_plip_interactions(pdb_file):
    """
    Retrieves the interactions from PLIP.

    Parameters
    ----------
    pdb_file :
        The PDB file of the complex.

    Returns
    -------
    dict :
        A dictionary of the binding sites and the interactions.
    """
    complex = PDBComplex()
    complex.load_pdb(pdb_file)
    for ligand in complex.ligands:
        complex.characterize_complex(ligand)

    # loop over binding sites
    sites = {}
    for key, site in sorted(complex.interaction_sets.items()):
        binding_site = BindingSiteReport(site)

        # tuples of *_features and *_info will be converted to pandas DataFrame
        keys = (
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        )

        # interactions is a dictionary which contains relevant information for each
        # of the possible interactions: hydrophobic, hbond, etc. in the considered
        # binding site. Each interaction contains a list with
        # 1. the features of that interaction, e.g. for hydrophobic:
        # ('RESNR', 'RESTYPE', ..., 'LIGCOO', 'PROTCOO')
        # 2. information for each of these features, e.g. for hydrophobic
        # (residue nb, residue type,..., ligand atom 3D coord., protein atom 3D coord.)
        interactions = {
            k: [getattr(binding_site, k + "_features")] + getattr(binding_site, k + "_info")
            for k in keys
        }
        sites[key] = interactions
    return sites

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

    # intersection
    common_elements_no= len(ref_set.intersection(var_set))

    if len(ref_set) == 0:
        return 0  # Edge case: both sets are empty

    # Calculate Tanimoto similarity
    tanimoto_similarity = common_elements_no / (len(ref_set) + len(var_df) - common_elements_no)

    return tanimoto_similarity

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

# write to an sdf
# with tempfile.TemporaryDirectory() as TD:
#     lig = parmed.load_file("structures/CC(C)(C)OC(=O)NC(=O)Sc1cccnc1.sdf")[0]
#     protein = parmed.load_file("../rec_final.pdb", structure=True)
#
#     system = protein + lig
#     complex_path = os.path.join(TD, "complex.pdb")
#     system.save(complex_path, renumber=False)
#
#     data = residue_interactions(complex_path, 0)
#
#     plip_dict = gen_intrns_dict(data)
#
#     data_interactions = set(plip_dict.keys())
#
#     plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE


# with tempfile.TemporaryDirectory() as TD:
#     lig = parmed.load_file("7l10_lig.sdf")[0]
#     protein = parmed.load_file("7l10_sup_prot.pdb")
#
#     system = protein + lig
#     complex_path = os.path.join(TD, "complex.pdb")
#     system.save(complex_path, renumber=False)
#
#     data = residue_interactions(complex_path, 0)
#     plip_dict = gen_intrns_dict(data)
#     data_interactions = set(plip_dict.keys())
#     plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE
#
# # confirm the previous selections
# data = residue_interactions("full_sess.pdb", 0)
# plip_dict = gen_intrns_dict(data)
# data_interactions = set(plip_dict.keys())
# plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE

# as the last exercise, run directly on the original input data:
# data = residue_interactions("5r83.pdb", 0)
# plip_dict = gen_intrns_dict(data)
# data_interactions = set(plip_dict.keys())
# plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE


# ok code it
# from plip.structure.preparation import PDBComplex
# my_mol = PDBComplex()
# my_mol.load_pdb(complex_path) # Load the PDB file into PLIP class
# print(my_mol) # Shows name of structure and ligand binding sites
# my_bsid = 'E20:A:2001' # Unique binding site identifier (HetID:Chain:Position)
# my_mol.analyze()
# assert len(my_mol.interaction_sets) == 1
#
# # take all the interactions
# all_interactions = list(my_mol.interaction_sets.values())[0]


def gen_plip_mpro_score(complex_path):
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


contacts = set()
for complex_path in Path("extracting_reference").glob("*pdb"):
    plip_dict = gen_intrns_dict(residue_interactions(str(complex_path)))

    plip_fp, tanimoto_distance = gen_plip_mpro_score(str(complex_path))
    print(plip_fp)
    print(tanimoto_distance)
    contacts = contacts.union(plip_dict)


# contacts = list(contacts)
# contacts.sort(key=lambda r: [int(r.split('_')[2]), r.split('_')[0], r.split('_')[1]])
# print(contacts)

# contacts = set()
# for complex_path in Path("extracting_reference").glob("*pdb"):
#     data = residue_interactions(str(complex_path), 0)
#     plip_dict = gen_intrns_dict(data)
#     data_interactions = set(plip_dict.keys())
#     print(data_interactions)
#     contacts = contacts.union(data_interactions)
#     # plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE
# contacts = list(contacts)
# contacts.sort(key=lambda r: [int(r.split('_')[2]), r.split('_')[0]])
# print('final')
# print(contacts)

