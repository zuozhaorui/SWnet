> # SWnet: a deep learning model for drug response prediction from cancer genomic signatures and compound chemical structures

Core code for the paper "SWnet: a deep learning model for drug response prediction from cancer genomic signatures and compound chemical structures" by Zhaorui Zuo, Penglei Wang, Xiaowei Chen, Li Tian, Hui Ge & Dahong Qian.

> ## Resources
---

### Data
The data in the folder is prepared for training and evaluating the SWnet.
* `data/GDSC/drug_similarity/GDSC_drug_similarity.csv`: This csv file record the similarity of drugs.
* `data/GDSC/GDSC_data`: The GDSC data which include 1478 genes across 1018 cell lines.
* `data/GDSC/graph_data`: The molecular graph information is saved in this data file.
* `data/CCLE/drug_similarity/CCLE_drug_similarity.csv`: This csv file record the similarity of drugs.
* `data/CCLE/CCLE_data`: The CCLE data which include 1478 genes across 469 cell lines.
* `data/CCLE/graph_data`: The molecular graph information is saved in this data file.

> ## Installation
---

Install the requirements (listed in environment.yaml). We're using Anaconda to install the environment:
```
conda create -f environment.yaml
conda activate swnet
pip install numpy==1.16.2
```

> ## Running the Code
---

### Model Code

As shown below, SWnet adopts a dual converge architercture.Genomic signature and chemical fingerprints are porcessed in parallel through GNN and CNN layers to extract independent features, which are then concatenated. And SWnet also integrate multi-task learning and self-attentation mechanism to further improve the performance.
The code for the SWnet can be found in `multi-task, self-attention, single-layer`.

### Evaluation on pretrained model
* `cd self-attention`
* `python SWnet_GDSC_self-attention_evaluate.py `
* `python SWnet_CCLE_self-attention_evaluate.py `
### or

### Train a prediction model on GDSC data
#### Prepare graph data, we can set the radius parameter to 1, 2, 3 or 4
* `cd data/GDSC` 
* `python preprocess_drug_graph.py --radius 1`

#### Prepare drug similarity data
* `cd data/GDSC`
* `python preprocess_drug_similarity.py`

#### Train self-attention SWnet 
* `cd self-attention`
* `python SWnet_GDSC_self-attention_train.py `

you can set hyper-parameter like this:
* `python SWnet_GDSC_self-attention_train.py --radius 3 --split_case 0 --layer_gnn 3`

#### Evaluate self-attention SWnet
* `cd self-attention`
* `python SWnet_GDSC_self-attention_evaluate.py `

### or

### Train a prediction model on CCLE data
#### Prepare graph data, we can set the radius parameter to 1, 2 ,3 or 4
* `cd data/CCLE` 
* `python preprocess_drug_graph.py --radius 1`

#### Prepare drug similarity data
* `cd data/CCLE`
* `python preprocess_drug_similarity.py`

#### Train self-attention SWnet 
* `cd self-attention`
* `python SWnet_CCLE_self-attention_train.py `

you can set hyper-parameter like this:
* `python SWnet_CCLE_self-attention_train.py --radius 3 --split_case 0 --layer_gnn 3`

#### Evaluate self-attention SWnet
* `cd self-attention`
* `python SWnet_CCLE_self-attention_evaluate.py `

#### Run Other scripts

The following scripts training the muti-task SWnet.
* `cd multi-task`
* `python SWnet_multi-task.py`

The following scripts training the single-layer SWnet.

* `cd single-layer`
* `python SWnet_single_no_weight.py`
* `python SWnet_single_yes_weight.py`

The following scripts training the GDSC gene weight Layer.

* `cd self-attention`
* `python SWnet_GDSC_self-attention_train.py --radius 3 --split_case 0`
* `python SWnet_CCLE_self-attention_train.py --radius 3 --split_case 0`

> ## Citation
---
If you find this code useful for your research, please use the following citation.
```
Zuo, Z., Wang, P., Chen, X. et al. SWnet: a deep learning model for drug response prediction from cancer genomic signatures and compound chemical structures. BMC Bioinformatics 22, 434 (2021). https://doi.org/10.1186/s12859-021-04352-9
```
