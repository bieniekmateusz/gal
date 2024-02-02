from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs




# generate fingerprints for each molecules
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

mols = list(Chem.SDMolSupplier("available_enamines.sdf"))
fps = [mfpgen.GetFingerprint(mol) for mol in mols]

n = len(mols)
dsts = np.zeros([n, n])

for i in range(n):
	for j in range(i+1, n):
		dsts[j][i] = dsts[i][j] = 1 - DataStructs.TanimotoSimilarity(fps[i],fps[j])
		
np.save("tanimoto_matrix.npy", dsts)