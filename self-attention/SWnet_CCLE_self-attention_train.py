import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import sys
sys.path.append('..')
import pickle
import argparse
import untils.until as untils
import random
# torch.manual_seed(0)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# graph dataset
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class CreateDataset(Dataset):
    def __init__(self, rma_all, var_all, id):
        self.rma_all = rma_all
        self.var_all = var_all
        self.all_id = id.values

    def __len__(self):
        return len(self.all_id)

    def __getitem__(self, idx):
        cell_line_id = self.all_id[idx][0]
        drug_id = self.all_id[idx][1]
        y = np.float32(self.all_id[idx][2])

        rma = self.rma_all.loc[cell_line_id].values.astype('float32')
        var = self.var_all.loc[cell_line_id].values.astype('float32')
        return rma, var, drug_id, y


class Model(nn.Module):
    def __init__(self,dim,layer_gnn,drugs_num):
        super(Model, self).__init__()
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(drugs_num, 1478), requires_grad=True).to(device)
        self.fuse_weight.data.normal_(0.5, 0.25)

        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])

        self.gene = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=15, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            # nn.Conv1d(in_channels=10, out_channels=10, kernel_size=30, stride=2),
            # nn.BatchNorm1d(10),
            # nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=15, stride=2),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Linear(71, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.merged = nn.Sequential(
            nn.Linear(210,100),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=10, stride=2),
            nn.BatchNorm1d(5),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=10, stride=2),
            nn.BatchNorm1d(5),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.1)
        )
        self.out = nn.Sequential(
            # nn.Linear(40, 20),
            # nn.Dropout(p=0.1),
            nn.Linear(5,1)
        )

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention(self,fuse_weight,drug_ids,var):
        try:
            drug_ids = drug_ids.numpy().tolist()
        except:
            drug_ids = drug_ids

        com = torch.zeros(len(drug_ids), 1478).to(device)

        for i in range(len(drug_ids)):
            com[i] = torch.mv(fuse_weight.permute(1, 0), similarity_softmax[GDSC_drug_dict[drug_ids[i]]]) * var[i]

        return com.view(-1,1478)

    def combine(self, rma, var,drug_id):
        self.fuse_weight.data = torch.clamp(self.fuse_weight, 0, 1)
        attention_var = self.attention(self.fuse_weight, drug_id, var)
        z = rma + attention_var
        return z

    def forward(self, rma, var, drug_id):
        com = self.combine(rma, var,drug_id)
        com = com.unsqueeze(1)
        out = self.gene(com)
        out_gene = out.view(out.size(0), -1)

        """Compound vector with GNN."""
        batch_graph = [graph_dataset[GDSC_drug_dict[i]] for i in drug_id]
        compound_vector = torch.FloatTensor(len(drug_id), dim).to(device)
        for i, graph in enumerate(batch_graph):
            fingerprints, adjacency = graph
            fingerprints.to(device)
            adjacency.to(device)
            fingerprint_vectors = self.embed_fingerprint(fingerprints)
            compound_vector[i] = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        # print(out.size(0),out.size(1))

        concat = torch.cat([out_gene, compound_vector], dim=1)
        # concat = concat.view(concat.size(0), -1)
        concat = concat.unsqueeze(1)

        merge = self.merged(concat)
        # print(merge.size(0), merge.size(1))
        merge = merge.view(merge.size(0), -1)

        y_pred = self.out(merge)
        return y_pred


def train_model(model, criterion, optimizer, scheduler, num_epochs=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        train_loss = 0.0
        model.train()
        for step, (rma, var, drug_id, y) in tqdm(enumerate(train_loader)):
            rma = rma.cuda(device=device_ids[0])
            var = var.cuda(device=device_ids[0])
            y = y.cuda(device=device_ids[0])
            y = y.view(-1, 1)
            # print('y',y)

            y_pred = model(rma, var, drug_id)
            # print('y_pred',y_pred)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rma.size(0)
        scheduler.step()

        test_loss = 0.0
        model.eval()
        for step, (rma, var, drug_id, y) in tqdm(enumerate(test_loader)):
            rma = rma.cuda(device=device_ids[0])
            var = var.cuda(device=device_ids[0])
            y = y.cuda(device=device_ids[0])
            y = y.view(-1, 1)

            y_pred = model(rma, var, drug_id)
            # print('y_pred',y_pred)
            loss = criterion(y_pred, y)
            test_loss += loss.item() * rma.size(0)

        epoch_train_loss = train_loss / dataset_sizes['train']
        epoch_test_loss = test_loss / dataset_sizes['test']

        print('Train Loss: {:.4f} Test Loss: {:.4f}'.format(epoch_train_loss, epoch_test_loss))
        log.write('Train Loss: {:.4f} Test Loss: {:.4f}\n'.format(epoch_train_loss, epoch_test_loss))
        log.flush()
        # deep copy the model
        if epoch_test_loss < best_loss and epoch>=3:
            best_loss = epoch_test_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.write('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(best_loss))
    log.write('Best test loss: {:4f}\n'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)

    pth_name = '../log/pth/' + str(round(best_loss,4)) + '_' + file_name + '_r' + str(radius) +'_s' + str(split_case) + '.pth'
    torch.save(model.state_dict(), pth_name)
    print("Save model done!")
    return model

def eval_model(model):
    from sklearn.metrics import r2_score, mean_squared_error
    y_pred = []
    y_true = []
    model.eval()
    for step, (rma, var, drug_id,y) in tqdm(enumerate(test_loader)):
        rma = rma.cuda(device=device_ids[0])
        var = var.cuda(device=device_ids[0])
        y = y.cuda(device=device_ids[0])
        y = y.view(-1, 1)
        # print('y',y)
        y_true += y.cpu().detach().numpy().tolist()
        y_pred_step = model(rma, var, drug_id)
        y_pred += y_pred_step.cpu().detach().numpy().tolist()
    return mean_squared_error(y_true, y_pred),r2_score(y_true, y_pred)


if __name__ == '__main__':

    """hyper-parameter"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--LR", type=float, default=0.001)
    parser.add_argument("--BATCH_SIZE", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=150)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--split_case", type=int, default=0)
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--layer_gnn", type=int, default=3)
    args = parser.parse_args()

    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE
    num_epochs = args.num_epochs
    step_size = args.step_size
    gamma = args.gamma
    split_case = args.split_case
    radius = args.radius
    dim = args.dim
    layer_gnn = args.layer_gnn
    """log"""
    dt = datetime.now()  # 创建一个datetime类对象
    file_name = os.path.basename(__file__)[:-3]
    date = dt.strftime('_%Y%m%d_%H_%M_%S')
    logname = '../log/logs/' + file_name +'_r' + str(radius) +'_s' + str(split_case)+ date + '.txt'
    logsaved = True
    if logsaved == True:
        log = open(logname, mode='wt')
    log.write(file_name + date + '.csv \n')
    log.write('radius = {:d},split case = {:d}\n'.format(radius, split_case))
    print("Log is start!")

    """Load preprocessed drug graph data."""
    dir_input = ('../data/CCLE/graph_data/' + 'radius' + str(radius) + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)

    """Create a dataset and split it into train/dev/test."""
    graph_dataset = list(zip(compounds, adjacencies))

    """Load CCLE data."""
    rma, var, smiles = untils.load_CCLE_data()

    smiles_vals = smiles["smiles"].values
    smiles_index = smiles.index
    cell_names = rma.index.values
    gene = rma.columns.values
    drugs_num = len(smiles_index)

    GDSC_drug_dict = untils.get_drug_dict(smiles)

    """Load CCLE drug similarity data."""

    data = pd.read_csv("../data/CCLE/drug_similarity/CCLE_drug_similarity.csv", header=None)
    similarity_softmax = torch.from_numpy(data.to_numpy().astype(np.float32))
    similarity_softmax = similarity_softmax.to(device)

    """split dataset"""
    data = pd.read_csv("../data/CCLE/CCLE_data/CCLE_cell_drug_labels.csv", index_col=0)
    train_id, test_id = untils.split_data(data,split_case=split_case, ratio=0.9,
                                   GDSC_cell_names=cell_names)

    dataset_sizes = {'train': len(train_id), 'test': len(test_id)}
    print(dataset_sizes['train'], dataset_sizes['test'])
    log.write('train size = {:d},test size = {:d}\n'.format(dataset_sizes['train'], dataset_sizes['test']))
    log.flush()

    trainDataset = CreateDataset(rma, var, train_id)
    testDataset = CreateDataset(rma, var, test_id)

    # Dataloader
    train_loader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE * len(device_ids), shuffle=True)
    test_loader = Data.DataLoader(dataset=testDataset, batch_size=BATCH_SIZE * len(device_ids), shuffle=True)

    """create SWnet model"""
    model_ft = Model(dim, layer_gnn, drugs_num)
    log.write(str(model_ft))

    """cuda"""
    model_ft = model_ft.cuda(device=device_ids[0])  #

    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    """start training model !"""
    print("start training model")

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=num_epochs)


    # model_ft.load_state_dict(torch.load("../log/pth/XXX.pth"))
    # mse, r2 = eval_model(model_ft)
    # print('mse:{},r2:{}'.format(mse, r2))

    mse, r2 = eval_model(model_ft)
    print('mse:{},r2:{}'.format(mse, r2))
    log.write('mse:{},r2:{}'.format(mse, r2))
    log.close()

    """Save the gene weights """
    fuse = pd.DataFrame(model_ft.fuse_weight.cpu().detach().numpy(),
                        index=smiles_index, columns=gene)

    fuse_name = '../log/gene_weights/' + str(round(mse, 4)) + '_' + file_name + '_r' + str(radius) + '_s' + str(
        split_case) + '.csv'
    fuse.to_csv(fuse_name)
    print("Save the gene weights done!")
