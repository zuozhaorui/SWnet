import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_drug_dict(smiles):
    try:
        drug_index = smiles.index.astype('float32')
    except:
        drug_index = smiles.index
    drug_dict = {}
    for i in range(len(drug_index)):
        drug_dict[drug_index[i]] = i
    return drug_dict

def split_data(data,split_case,ratio,cell_names):
    data = data[data["labels"].notnull()]
    data = data[~data['drug_id'].isin([185, 1021])]  # except drug id
    # Split data sets randomly
    if split_case == 0:
        train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)

    # split data sets by cells
    elif split_case == 1:
        np.random.seed(0)
        np.random.shuffle(cell_names)
        n = int(ratio * len(cell_names))
        train_id = data[data['cell_line_id'].isin(cell_names[:n])]
        test_id = data[data['cell_line_id'].isin(cell_names[n:])]
    # all data sets
    elif split_case == 2:
        train_id = data
        _ , test_id = train_test_split(data, test_size=1 - ratio, random_state=0)

    # train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)
    # train_id, test_id = train_id[0:100], test_id[0:100]
    return train_id, test_id

def load_GDSC_data():
    GDSC_rma_path = "../data/GDSC/GDSC_data/GDSC_rma.csv"
    GDSC_variant_path = "../data/GDSC/GDSC_data/GDSC_variant.csv"
    GDSC_smiles_path = "../data/GDSC/GDSC_data/GDSC_smiles.csv"

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles

def load_CCLE_data():
    GDSC_rma_path = "../data/CCLE/CCLE_data/CCLE_RNAseq.csv"
    GDSC_variant_path = "../data/CCLE/CCLE_data/CCLE_DepMap.csv"
    GDSC_smiles_path = "../data/CCLE/CCLE_data/CCLE_smiles.csv"

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles