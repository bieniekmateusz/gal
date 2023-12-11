"""
This is a script to be used after FEgrow was used to build the chemical space (dock and sccore).

Extract from .sdf files the scores into a pandas dataframe.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

cs = pd.read_csv('manual_init_h6_rgroups_linkers100.csv', usecols=["Smiles"])

cs["cnnaffinity"] = np.nan
cnnaffinity_col_index = cs.columns.get_loc('cnnaffinity')

for sdf in Path(".").glob("scored/*sdf"):
    sdf_id = int(sdf.stem)
    mol = Chem.SDMolSupplier(str(sdf))[0]
    cnnaffinity = float(mol.GetProp('cnnaffinity'))

    cs.iloc[sdf_id, cnnaffinity_col_index ] = cnnaffinity

cs.dropna(inplace=True)

cs.to_csv("cs_scored.csv", index_label='fid')

# also update the previous array as to which are scorable and which are not
cs_full = pd.read_csv('manual_init_h6_rgroups_linkers100.csv', index_col='fid')
cs_scorable = cs_full[cs_full.index.isin(cs.index)]
cs_scorable.to_csv('manual_init_h6_rgroups_linkers100_scorable.csv')
print('hi')