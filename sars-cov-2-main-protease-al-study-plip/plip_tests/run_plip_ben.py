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
        print("!!! Wrong interaction type specified. Hbond is chosen by default!!!\n")
        interaction_type = "hbond"

    df = pd.DataFrame.from_records(
        # data is stored AFTER the column names
        selected_site_interactions[interaction_type][1:],
        # column names are always the first element
        columns=selected_site_interactions[interaction_type][0],
    )
    return df

def residue_interactions(pdb_id, site):
    interactions_by_site = retrieve_plip_interactions(f"{pdb_id}")
    index_of_selected_site = 0
    selected_site = list(interactions_by_site.keys())[index_of_selected_site]
    # print(selected_site)
    dfs = []
    valid_types = [
        #            "hydrophobic",
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
    print(all_interactions)
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
    protlig = PDBComplex()
    protlig.load_pdb(pdb_file)  # load the pdb file
    for ligand in protlig.ligands:
        protlig.characterize_complex(ligand)  # find ligands and analyze interactions
    sites = {}
    # loop over binding sites
    for key, site in sorted(protlig.interaction_sets.items()):
        binding_site = BindingSiteReport(site)  # collect data about interactions
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
xstal_interaction_list = [   'hbond_ASN_142_O2',
                             'hbond_ASN_142_O3',
                             'hbond_CYS_145_O3', #
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

# write to an sdf
with tempfile.TemporaryDirectory() as TD:
    lig = parmed.load_file("structures/CC(C)(C)OC(=O)NC(=O)Sc1cccnc1.sdf")[0]
    protein = parmed.load_file("../rec_final.pdb", structure=True)

    system = protein + lig
    complex_path = os.path.join(TD, "complex.pdb")
    system.save(complex_path, renumber=False)

    data = residue_interactions(complex_path, 0)

    plip_dict = gen_intrns_dict(data)

    data_interactions = set(plip_dict.keys())

    plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE


with tempfile.TemporaryDirectory() as TD:
    lig = parmed.load_file("7l10_lig.sdf")[0]
    protein = parmed.load_file("7l10_sup_prot.pdb")

    system = protein + lig
    complex_path = os.path.join(TD, "complex.pdb")
    system.save(complex_path, renumber=False)

    data = residue_interactions(complex_path, 0)

    plip_dict = gen_intrns_dict(data)

    data_interactions = set(plip_dict.keys())

    plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE


# as the last exercise, run directly on the original input data:
data = residue_interactions("5r83.pdb", 0)
plip_dict = gen_intrns_dict(data)
data_interactions = set(plip_dict.keys())
plip_score(xstal_set, data_interactions) * 50  # HYPERPARAM TO CHANGE