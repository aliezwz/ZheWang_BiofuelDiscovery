import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict

import deepchem as dc
import deepchem.models
import torch
from deepchem.models.torch_models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
from torch.nn.functional import one_hot
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import time

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

def print_time(start, end, task_name):
    print(f"{task_name} took {end - start:.2f} seconds")

# Start timing data loading
start_time = time.time()

# Read SMILES data
start_time = time.time()
df = pd.read_csv("../Dataset/Smiles list.csv")
end_time = time.time()
print_time(start_time, end_time, "Reading SMILES list")

num_atoms = 12

data = df

# create featurizer
feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14]) #15, 16, 17, 19, 20, 24, 29, 35, 53, 80])

smiles = data['SMILES'].values

filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]

# featurize molecules
features = feat.featurize(filtered_smiles)

indices = [ i for i, data in enumerate(features) if type(data) is GraphMatrix ]
print(indices)
features = [features[i] for i in indices]

# create model
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],[x.node_features for x in features])

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            flattened_adjacency = torch.from_numpy(batch[0]).view(-1).to(dtype=torch.int64) # flatten the input because torch.nn.functional.one_hot only works with 1D inputs
            invalid_mask = (flattened_adjacency < 0) | (flattened_adjacency >= gan.edges) # edge type cannot be negative or >= gan.edges, these entries are invalid
            clamped_adjacency = torch.clamp(flattened_adjacency, 0, gan.edges-1) # clamp the input so it can be fed to the one_hot function
            adjacency_tensor = one_hot(clamped_adjacency, num_classes=gan.edges) # actual one_hot
            adjacency_tensor[invalid_mask] = torch.zeros(gan.edges, dtype=torch.long) # make the invalid entries, a vector of zeros
            adjacency_tensor = adjacency_tensor.view(*batch[0].shape, -1) # reshape to original shape and change dtype for stability.

            flattened_node = torch.from_numpy(batch[1]).view(-1).to(dtype=torch.int64)
            invalid_mask = (flattened_node < 0) | (flattened_node >= gan.nodes)
            clamped_node = torch.clamp(flattened_node, 0, gan.nodes-1)
            node_tensor = one_hot(clamped_node, num_classes=gan.nodes)
            node_tensor[invalid_mask] = torch.zeros(gan.nodes, dtype=torch.long)
            node_tensor = node_tensor.view(*batch[1].shape, -1)

            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}

gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)
generated_data = gan.predict_gan_generator(10000)

nmols = feat.defeaturize(generated_data)
print("{} molecules generated".format(len(nmols)))

nmols = list(filter(lambda x: x is not None, nmols))

# currently training is unstable so 0 is a common outcome
print ("{} valid molecules".format(len(nmols)))

nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print ("{} unique valid molecules".format(len(nmols_viz)))

# Print Generated SMILES
print("Generated SMILES:")
for smiles in nmols_smiles_unique:
    print(smiles)