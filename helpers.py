import os
from typing import NamedTuple, List
from rdkit import Chem
import dataclasses
from collections import Counter


import re
import pandas as pd

import uuid


@dataclasses.dataclass
class Data():
    filename: str = None
    # ie [hid, tried, successfull]
    hydrogens: List[int] = None
    cnnaffinity: float = None
    cnnaffinityIC50: float = None
    MW: float = None
    HBA: int = None
    HBD: int = None
    LogP: float = None
    Pass_Ro5: bool = None
    has_pains: bool = None
    has_unwanted_subs: bool = None
    has_prob_fgs: bool = None
    synthetic_accessibility: float = None
    Smiles: str = None
    QED: float = None
    interactions: list = None

    def __repr__(self):
        return f"{self.filename} {self.hydrogens} {self.cnnaffinity} {self.cnnaffinityIC50}"


def set_properties(mol, data):
    if data.filename == None:
        data.filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

    # add the data as properties to the file
    for field in dataclasses.fields(Data):
        mol.SetProp(field.name, str(getattr(data, field.name)))


def save(mol):
    filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

    if os.path.exists(f'structures/{filename}.sdf'):
        print('Overwriting the existing file')
    else:
        print(f'Creating a new file: {filename}.sdf')

    with Chem.SDWriter(f'structures/{filename}.sdf') as SD:
        # add the data as properties to the file
        SD.write(mol)
        print(f'Saved {filename}')


def save_all(items):
    for mol, data in items:
        set_properties(mol, data)
        save(mol)


def data(mol):
    # convert all properties into data
    data = Data()
    for prop, value in mol.GetPropsAsDict().items():
        # avoid evaluating smiles
        if str(prop).strip() in ['Smiles', 'filename']:
            setattr(data, prop, value)
            continue
        else:
            setattr(data, prop, eval(str(value)))
    return data
    # hstr = mol.GetProp('hydrogens')
    # if not hstr.strip():
    # 	return {}

    # hs = hstr.split(',')

    # hcounter = []
    # for hydrogen_data in hs:
    # 	hydrogen, tried, successful =  hydrogen_data.split(':')
    # 	hcounter.append([int(hydrogen),int(tried), int(successful)])

    # return hcounter
def scale_df(df):
    '''Scale the metrics to [0, 1] for columns containing floats'''
    if type == 'minmax':
        scaler = MinMaxScaler()
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = scaler.fit_transform(df[float_cols])
        return df

def add_sf(file1, file2, func, scale=True, display_name=None, *args):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if set(df1.columns) != set(df2.columns):
        mismatched_columns = set(df1.columns).symmetric_difference(set(df2.columns))
        raise ValueError(f"Mismatched columns: {mismatched_columns}")

    func_name = func.__name__ if display_name is None else display_name

    if scale:
        scaled_df1 = scale_df(df1.copy())
        scaled_df2 = scale_df(df2.copy())

        df1[func_name] = func(scaled_df1)
        df2[func_name] = func(scaled_df2)
    else:
        df1[func_name] = func(df1)
        df2[func_name] = func(df2)

    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    return df1, df2




def exp_qed_cnn(df):
    '''10^cnn / mw * qed'''
    col_name = 'exp_cnn_per_mw'
    if col_name not in df.columns or 'QED' not in df.columns:
        raise KeyError(f"Missing required columns: {col_name} and/or 'cnn_per_mw'")
    return df['exp_cnn_per_mw'] * df['QED']

def gen_intrns_dict(interactions):
    """
    Generate an interaction dictionary sorted by residue number and secondarily by residue name.

    Parameters:
    interactions (list): List of interaction strings

    Returns:
    dict: Sorted interaction dictionary mapping interaction strings to bit positions
    """
    # extract RESNR and RESNAME from interactions using regex
    interaction_tuples = []
    for interaction in interactions:
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



def encode_intrns(interactions, interaction_dict):
    """
    Encode a list of interaction strings to a bit vector using a predefined interaction dictionary.

    Parameters:
    interactions (list): List of interaction strings
    interaction_dict (dict): Dictionary mapping interaction strings to bit positions

    Returns:
    list: Encoded bit vector

    # Example usage
    interaction_dict = {
    'hbond_CYS_143_N1': 0,
    'hbond_SER_142_N1': 1,
    'hbond_GLY_141_N1': 2,
    'pistacking_HIS_41': 3,
    'hbond_GLU_164_O2': 4
    }

    interactions1 = ['hbond_CYS_143_N1', 'hbond_SER_142_N1']
    bit_vector1 = encode_interactions_to_bitvector(interactions1, interaction_dict)

    """
    bit_vector = [0] * len(interaction_dict)

    for interaction in interactions:
        if interaction in interaction_dict:
            bit_position = interaction_dict[interaction]
            bit_vector[bit_position] = 1

    return bit_vector




import pandas as pd

def plip_tanimoto(ref_df, var_df):
    """
    Compute Tanimoto similarity between a reference df of xstal plip interactions & another molecule.

    Parameters:
    ref_df (DataFrame): Reference DataFrame
    var_df (DataFrame): Variable DataFrame to compare

    Returns:
    float: Tanimoto similarity score
    """
    assert len(ref_df.columns) == 1 and len(var_df.columns) == 1, "DataFrames should have only one column"

    ref_set = set(ref_df.iloc[:, 0])
    var_set = set(var_df.iloc[:, 0])

    intersection_count = len(ref_set.intersection(var_set))
    union_count = len(ref_set.union(var_set))

    if union_count == 0:
        return 0  # Edge case: both sets are empty

    return intersection_count / union_count







xstal_interaction_list = ['hbond_THR_25_N3',
 'hbond_HIS_41_O2',
 'hbond_CYS_44_O2',
 'pistacking_HIS_41_nan',
 'hbond_SER_144_O3',
 'hbond_GLU_166_O3',
 'hbond_SER_144_O3',
 'hbond_GLU_166_O2',
 'hbond_GLU_166_O3',
 'hbond_GLU_166_O2',
 'pistacking_HIS_41_nan',
 'hbond_GLU_166_O2',
 'hbond_THR_25_N3',
 'hbond_HIS_41_O2',
 'hbond_ASN_142_O2',
 'hbond_ASN_142_O3',
 'hbond_GLU_166_N3',
 'hbond_SER_144_O3',
 'hbond_GLU_166_O3',
 'hbond_ASN_142_O2',
 'pistacking_HIS_41_nan',
 'hbond_GLN_189_O2',
 'hbond_GLU_166_O3',
 'hbond_HIS_163_N2',
 'hbond_GLN_189_O3',
 'pistacking_HIS_41_nan',
 'hbond_GLU_166_O3',
 'saltbridge_GLU_166_nan',
 'hbond_GLU_166_O.co2',
 'hbond_GLU_166_O3',
 'hbond_GLU_166_O2',
 'hbond_ASN_142_O2',
 'hbond_GLY_143_O3',
 'hbond_SER_144_O3',
 'hbond_CYS_145_O3',
 'hbond_HIS_163_N2',
 'hbond_GLU_166_N3',
 'hbond_CYS_44_O2',
 'hbond_GLU_166_O2',
 'hbond_GLU_166_O2']

xstal_set = set(xstal_interaction_list)

frag_7l10 = ['hbond_GLU_166_O3',
 'hbond_GLY_143_Nox',
 'hbond_CYS_145_Nox',
 'hbond_HIS_163_N2',
 'hbond_GLU_166_O2']

ref_df = pd.DataFrame(xstal_set)
var_df = pd.DataFrame(frag_7l10)
