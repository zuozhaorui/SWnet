import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_drug_dict(GDSC_smiles_index):
    drug_index = GDSC_smiles_index.astype('float32')
    drug_dict = {}
    for i in range(len(drug_index)):
        drug_dict[drug_index[i]] = i
    return drug_dict

def split_data(split_case,ratio,GDSC_cell_names):

    data = pd.read_csv("../data/GDSC_data/cell_drug_labels.csv", index_col=0)
    data = data[data["labels"].notnull()]
    data = data[~data['drug_id'].isin([185, 1021])]  # except drug id
    # Split data sets randomly
    if split_case == 0:
        train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)

    # split data sets by cells
    elif split_case == 1:
        np.random.seed(0)
        np.random.shuffle(GDSC_cell_names)
        n = int(ratio * len(GDSC_cell_names))
        train_id = data[data['cell_line_id'].isin(GDSC_cell_names[:n])]
        test_id = data[data['cell_line_id'].isin(GDSC_cell_names[n:])]
    # all data sets
    elif split_case == 2:
        train_id = data
        _ , test_id = train_test_split(data, test_size=1 - ratio, random_state=0)

    # train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)
    # train_id, test_id = train_id[0:100], test_id[0:100]
    return train_id, test_id

def load_GDSC_data():
    GDSC_rma_path = "../data/GDSC_data/GDSC_rma.csv"
    GDSC_variant_path = "../data/GDSC_data/GDSC_variant.csv"
    GDSC_smiles_path = "../data/GDSC_data/GDSC_smiles.csv"

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles