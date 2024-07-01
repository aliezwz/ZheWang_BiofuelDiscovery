import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import deepchem as dc
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit import Chem
from rdkit.Chem import Draw


# Function to print timing information
def print_time(start, end, task_name):
    print(f"{task_name} took {end - start:.2f} seconds")


# Function to read SMILES data
def read_smiles(file_path):
    df = pd.read_csv(file_path)
    return df['SMILES'].values


# Function to create featurizer
def create_featurizer(max_atom_count):
    return dc.feat.MolGanFeaturizer(max_atom_count=max_atom_count,
                                    atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])


# Function to filter SMILES
def filter_smiles(smiles, max_num_atoms):
    return [x for x in smiles if Chem.MolFromSmiles(x) and Chem.MolFromSmiles(x).GetNumAtoms() < max_num_atoms]


# Function to featurize molecules
def featurize_molecules(featurizer, smiles):
    features = featurizer.featurize(smiles)
    return [feat for feat in features if isinstance(feat, dc.feat.molecule_featurizers.molgan_featurizer.GraphMatrix)]


# Function to create and prepare GAN model
def create_gan_model(learning_rate, num_atoms):
    gan = MolGAN(learning_rate=learning_rate, vertices=num_atoms)
    return gan


# Function to prepare dataset
def prepare_dataset(features):
    adjacency_matrices = [x.adjacency_matrix for x in features]
    node_features = [x.node_features for x in features]
    return dc.data.NumpyDataset(adjacency_matrices, node_features)


# Function to generate batches
def iterbatches(dataset, gan, epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}


# Function to train GAN
def train_gan(gan, dataset, epochs, generator_steps, checkpoint_interval):
    gan.fit_gan(iterbatches(dataset, gan, epochs), generator_steps=generator_steps, checkpoint_interval=checkpoint_interval)


# Function to generate data
def generate_data(gan, num_samples):
    return gan.predict_gan_generator(num_samples)


# Function to defeaturize generated data
def defeaturize_generated_data(featurizer, generated_data):
    return featurizer.defeaturize(generated_data)


# Function to save generated SMILES
def save_generated_smiles(smiles, output_dir, iteration):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'generated_smiles_iteration_{iteration}.txt')
    with open(file_path, 'w') as f:
        for s in smiles:
            f.write(f"{s}\n")


# Function to save molecule images
def save_molecule_images(molecules, output_dir, iteration):
    img = Draw.MolsToGridImage(molecules[:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100)
    file_path = os.path.join(output_dir, f'generated_molecules_iteration_{iteration}.png')
    img.save(file_path)


# Function to save metrics
def save_metrics(metrics, output_dir, iteration):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'metrics_iteration_{iteration}.txt')
    with open(file_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")


# Main function to control GAN training and generation
def main(file_path, output_dir, num_atoms=12, epochs=25, iterations=10, num_samples=10000, generator_steps=0.2, checkpoint_interval=5000):
    start_time = time.time()

    smiles = read_smiles(file_path)
    featurizer = create_featurizer(num_atoms)
    filtered_smiles = filter_smiles(smiles, num_atoms)
    features = featurize_molecules(featurizer, filtered_smiles)

    learning_rate = ExponentialDecay(0.001, 0.9, 5000)
    gan = create_gan_model(learning_rate, num_atoms)
    dataset = prepare_dataset(features)

    for iteration in range(iterations):
        print(f"Starting iteration {iteration + 1}/{iterations}")
        iter_start_time = time.time()

        train_gan(gan, dataset, epochs, generator_steps, checkpoint_interval)

        generated_data = generate_data(gan, num_samples)
        nmols = defeaturize_generated_data(featurizer, generated_data)

        nmols = list(filter(lambda x: x is not None, nmols))
        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
        nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]

        iteration_output_dir = os.path.join(output_dir, f'iteration_{iteration + 1}')

        save_generated_smiles(nmols_smiles_unique, iteration_output_dir, iteration + 1)
        #save_molecule_images(nmols_viz, iteration_output_dir, iteration + 1)

        metrics = {
            "total_molecules": len(nmols),
            "unique_molecules": len(nmols_smiles_unique),
            "iteration_time": time.time() - iter_start_time
        }
        save_metrics(metrics, iteration_output_dir, iteration + 1)

        print(f"Iteration {iteration + 1}/{iterations} completed")

    end_time = time.time()
    print_time(start_time, end_time, "Total process")


# External script control function
def run_gan_training():
    file_path = "../Dataset/Smiles list.csv"
    output_dir = "output_dir"
    num_atoms = 12
    epochs = 100
    iterations = 2
    num_samples = 1000
    generator_steps = 0.2
    checkpoint_interval = 500

    main(file_path, output_dir, num_atoms, epochs, iterations, num_samples, generator_steps, checkpoint_interval)


if __name__ == "__main__":
    run_gan_training()
