import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
import torch
import pandas as pd

def drug_similarity(smiles):
    softmax_fun = torch.nn.Softmax(dim=1)
    N = len(smiles)
    simi = np.zeros((N,N))
    for i in range(N):
        col = []
        for j in range(N):
            ms = [Chem.MolFromSmiles(smiles[i]),Chem.MolFromSmiles(smiles[j])]
            fps = [FingerprintMols.FingerprintMol(x) for x in ms]
            similarity = DataStructs.FingerprintSimilarity(fps[0], fps[1])
            col.append(similarity)
        simi[i] = col

    input = torch.from_numpy(simi.astype(np.float32))
    similarity_softmax = softmax_fun(input)

    return similarity_softmax

if __name__ == "__main__":
    GDSC_smiles_path = "GDSC_data/GDSC_smiles.csv"
    GDSC_smiles = pd.read_csv(GDSC_smiles_path, index_col=0)
    GDSC_smiles_vals = GDSC_smiles["smiles"].values

    similarity_softmax = drug_similarity(GDSC_smiles_vals)
    GDSC_softmax_similarity = pd.DataFrame(similarity_softmax.numpy(), columns=None, index=None)
    GDSC_softmax_similarity.to_csv("drug_similarity/GDSC_drug_similarity.csv",header=None,index=None)

    print("GDSC drug similarity has finished!")