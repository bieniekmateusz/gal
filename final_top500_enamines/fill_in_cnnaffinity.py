from rdkit import Chem
import fegrow

fegrow.RMol.set_gnina('/home/dresio/gnina')


mols = list(Chem.SDMolSupplier("available_enamines.sdf"))
for mol in mols:
	props = mol.GetPropsAsDict()
	if 'cnnaffinity' in props:
		continue
	
	rmol = fegrow.RMol(mol)

	df = rmol.gnina("/home/dresio/code/al_for_fep/sars-cov-2-main-protease-al-study-combo1/rec_final.pdb")
	mol.SetProp("cnnaffinity", str(-df.iloc[0].CNNaffinity))

with Chem.SDWriter("available_enamines_cnnaffinity.sdf") as SD:
	for mol in mols:
		SD.write(mol)