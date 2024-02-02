from rdkit import Chem

mols = list(Chem.SDMolSupplier("available_enamines_cnnaffinity.sdf"))
for mol in mols:
	cnn = float(mol.GetProp("cnnaffinity"))
	if cnn < 0:
		cnn = -cnn
		mol.SetProp("cnnaffinity", str(cnn))

with Chem.SDWriter("available_enamines_cnnaffinity_sign.sdf") as SD:
	for mol in mols:
		SD.write(mol)