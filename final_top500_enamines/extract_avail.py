from pathlib import Path

from rdkit import Chem


# get the info if the molecules are available in the Enamine database
lines = [l.strip() for l in open("availability.log").readlines()]
available_enamines = [l.split()[0] for l in filter(lambda l: 'found' in l, lines)]

mols = []
for sdf in Path(".").glob("*sdf"):
	for mol in Chem.SDMolSupplier(str(sdf)):
		# check if the molecule is available in Enamine
		if mol.GetProp("enamine_id") in available_enamines:
			mol.SetProp("study", str(sdf))
			mols.append(mol)

with Chem.SDWriter("available_enamines.sdf") as SD:
	for mol in mols:
		SD.write(mol)