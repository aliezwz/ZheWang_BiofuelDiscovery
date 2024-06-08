import deepchem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict
import time

import deepchem as dc
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

# Timing function
def print_time(start, end, task_name):
    print(f"{task_name} took {end - start:.2f} seconds")

# Start timing data loading
start_time = time.time()

# Read SMILES data
start_time = time.time()
df = pd.read_csv("../Dataset/Smiles list.csv")
end_time = time.time()
print_time(start_time, end_time, "Reading SMILES list")

num_atoms = 15
print(df)

# data = df[['smiles']].sample(4000, random_state=42)
data = df

# Create featurizer
start_time = time.time()
feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])
end_time = time.time()
print_time(start_time, end_time, "Creating featurizer")

smiles = data['SMILES'].values

start_time = time.time()
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]
end_time = time.time()
print_time(start_time, end_time, "Filtering SMILES")

# Featurize molecules
start_time = time.time()
features = feat.featurize(filtered_smiles)
end_time = time.time()
print_time(start_time, end_time, "Featurizing molecules")

indices = [i for i, data in enumerate(features) if type(data) is GraphMatrix]
print(indices)
features = [features[i] for i in indices]

# Create model
start_time = time.time()
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])
end_time = time.time()
print_time(start_time, end_time, "Creating model and dataset")

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}

# Train GAN
gan.fit_gan(iterbatches(25), generator_steps=0.7, checkpoint_interval=5000)

# Generate data
start_time = time.time()
generated_data = gan.predict_gan_generator(10000)
end_time = time.time()
print_time(start_time, end_time, "Generating data")
time.sleep(10)

start_time = time.time()
nmols = feat.defeaturize(generated_data)
end_time = time.time()
print_time(start_time, end_time, "Defeaturizing generated data")
print("{} molecules generated".format(len(nmols)))

nmols = list(filter(lambda x: x is not None, nmols))

print("{} valid molecules".format(len(nmols)))

# Initialize list to save SMILES
generated_smiles_list = []

nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print("{} unique valid molecules".format(len(nmols_viz)))

# Add generated SMILES to the list
generated_smiles_list.extend(nmols_smiles_unique)

# Print generated SMILES
print("Generated SMILES:")
for smiles in nmols_smiles_unique:
    print(smiles)

# Save generated SMILES to file
with open('generated_smiles_epoches=100.txt', 'w') as f:
    for smiles in generated_smiles_list:
        f.write(f"{smiles}\n")

# Generate image
start_time = time.time()
img = Draw.MolsToGridImage(nmols_viz[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None, returnPNG=False)
end_time = time.time()
print_time(start_time, end_time, "Generating image")

# Display image using matplotlib
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
