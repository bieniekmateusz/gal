import os
from typing import NamedTuple, List
from rdkit import Chem
import dataclasses

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
