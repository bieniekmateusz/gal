import warnings
import tempfile
import os
from pathlib import Path

import parmed
from plip.structure.preparation import PDBComplex
from rdkit import Chem


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


def plip_mpro_merge_score(protein, ligand):
    with tempfile.TemporaryDirectory() as TD:

        # turn the rdkit.Mol into an SDF file
        if isinstance(ligand, Chem.Mol):
            ligand_path = os.path.join(TD, "ligand.sdf")
            with Chem.SDWriter(ligand_path) as SD:
                SD.write(ligand)
                ligand = ligand_path

        lig = parmed.load_file(ligand)
        if isinstance(lig, list):
            warnings.warn("The ligand was an array (SDF?). Using the first frame. ")
            lig = lig[0]
        protein = parmed.load_file(protein)
        system = protein + lig
        complex_path = os.path.join(TD, "complex.pdb")
        system.save(complex_path, renumber=False)
        return plip_mpro_score(str(complex_path))

def plip_mpro_score(complex_path):
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


if __name__ == "__main__":
    ligand = Chem.SDMolSupplier("7l10_lig.sdf")[0]
    fp, tanimoto_score = plip_mpro_merge_score("7l10_sup_prot.pdb", ligand)

    contacts = set()
    for complex_path in Path("extracting_reference").glob("*pdb"):
        plip_fp, tanimoto_distance = plip_mpro_score(str(complex_path))
        contacts = contacts.union(set(plip_fp))

    contacts = list(contacts)
    contacts.sort(key=lambda r: [int(r.split('_')[2]), r.split('_')[0], r.split('_')[1]])
    print(contacts)

