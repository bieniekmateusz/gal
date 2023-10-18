import os
from typing import NamedTuple, List
from rdkit import Chem
import dataclasses
from collections import Counter

import numpy as np
import re
import pandas as pd

import uuid
cwd = os.getcwd()

@dataclasses.dataclass
class Data():
    ''' mol data, and scoring functions, scoring functions (sf1, sf2, ..) are synthetically created from data '''
    filename: str = None
    # ie [hid, tried, successfull]
    hydrogens: List[int] = None
    cnnaffinity: float = None
    cnnaffinityIC50: float = None
    cnn_ic50_norm: float = None
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
    interactions: set = None
    plip: float = 1
    cycle: int = None
    sf1: float = None
    sf2: float = None

    def __repr__(self):
        return f"{self.filename} {self.hydrogens} {self.cnnaffinity} {self.cnnaffinityIC50} {self.sf1}"

def sf1(rmol_data):
    # assert that rmol_data contains the necessary data
    try:
        assert rmol_data.plip is not None, "plip attribute is missing or None"
        assert rmol_data.interactions is not None, "interactions attribute is missing or None"
        assert rmol_data.cnn_ic50_norm is not None, "cnn_ic50_norm is missing or None"
        assert rmol_data.MW is not None, "MW is missing or None"
    except AssertionError as e:
        raise ValueError(f"Missing necessary data: {e}")
    sf1 = (rmol_data.QED * rmol_data.plip * (1 + len(rmol_data.interactions))) * (10**(rmol_data.cnnaffinity) / rmol_data.MW) 
    return sf1


    
def set_properties(mol, data):
    if data.filename == None:
        data.filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

    # add the data as properties to the file
    for field in dataclasses.fields(Data):
        mol.SetProp(field.name, str(getattr(data, field.name)))


def save(mol):
    print(f'cwd = {cwd}')
    filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

    if os.path.exists(f'{cwd}/structures/{filename}.sdf'):
        print('Overwriting the existing file')
    else:
        print(f'Creating a new file: {filename}.sdf')

    with Chem.SDWriter(f'{cwd}/structures/{filename}.sdf') as SD:
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
        if str(prop).strip() in ['Smiles', 'filename', 'plip']:
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
    col_name = 'cnnaff'
    if col_name not in df.columns or 'cnn_per_mw' not in df.columns:
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


#bit_vector2 = encode_interactions_to_bitvector(interactions2, interaction_dict)

#bit_vector1, bit_vector2



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

    # Compute intersection
    common_elements = len(ref_set.intersection(var_set))

    # Compute the total number of elements in the reference set
    total_elements = len(ref_set)

    if total_elements == 0:
        return 0  # Edge case: both sets are empty

    # Calculate Tanimoto similarity
    tanimoto_similarity = common_elements / total_elements

    return tanimoto_similarity


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
