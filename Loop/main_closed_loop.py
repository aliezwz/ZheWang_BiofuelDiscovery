import os
import time
import pandas as pd
from rdkit import Chem
from collections import OrderedDict
from deepchem.models.optimizers import ExponentialDecay
from RF_classifier import (
    read_data,
    classifier_core,
)
from MolGAN_loop import (
    create_featurizer,
    filter_smiles,
    featurize_molecules,
    create_gan_model,
    prepare_dataset,
    train_gan,
    generate_data,
    defeaturize_generated_data,
    save_generated_smiles,
    save_metrics,
)
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def measure_molecular_diversity(smiles_list, bins=10):
    # Convert SMILES to molecular representations
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Calculate molecular fingerprints
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]

    # Calculate pairwise Tanimoto similarities
    def calculate_similarity_matrix(fingerprints):
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        return similarity_matrix

    similarity_matrix = calculate_similarity_matrix(fingerprints)

    # Compute Shannon Entropy
    def compute_shannon_entropy(similarity_matrix, bins=10):
        similarities = similarity_matrix[np.triu_indices(len(similarity_matrix), 1)]
        histogram, _ = np.histogram(similarities, bins=bins, range=(0, 1), density=True)
        histogram = histogram[histogram > 0]
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

    shannon_entropy = compute_shannon_entropy(similarity_matrix, bins=bins)

    # Calculate molecular descriptors
    descriptors = np.array([rdMolDescriptors.CalcExactMolWt(mol) for mol in molecules]).reshape(-1, 1)

    # Compute centroid of descriptors
    centroid = np.mean(descriptors, axis=0)

    # Compute mean distance to centroid
    mean_distance_to_centroid = np.mean([np.linalg.norm(descriptor - centroid) for descriptor in descriptors])

    # Results
    results = {
        "Shannon Entropy": shannon_entropy,
        "Mean Distance to Centroid": mean_distance_to_centroid
    }

    return results, similarity_matrix, descriptors

def plot_similarity_distribution(similarity_matrix, output_dir, bins=10):
    similarities = similarity_matrix[np.triu_indices(len(similarity_matrix), 1)]
    sns.histplot(similarities, bins=bins, kde=True)
    plt.title('Distribution of Pairwise Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    plt.close()

def plot_descriptor_distribution(descriptors, output_dir, bins=10):
    sns.histplot(descriptors, bins=bins, kde=True)
    plt.title('Distribution of Molecular Descriptors')
    plt.xlabel('Descriptor Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'descriptor_distribution.png'))
    plt.close()

def main_closed_loop():
    # Configuration
    rf_csv_file = 'Dataset_edit.csv'
    lotus_csv = 'Lotus_dataset_mini.csv'
    molgan_output_dir = 'MolGAN_output'
    num_iterations = 5
    gan_params = {
        'num_atoms': 15,
        'epochs': 2000,
        'num_samples': 100000,
        'generator_steps': 0.2,
        'checkpoint_interval': 5000,
    }
    confidence_threshold = 0.66

    df = read_data(rf_csv_file)
    lotus_and_generated_smiles = pd.read_csv(lotus_csv)

    diversity_over_iterations = []
    classifier_performance_over_iterations = []
    molgan_performance_over_iterations = []

    start_time = time.time()
    for iteration in range(1, num_iterations + 1):
        print(f"Starting iteration {iteration}/{num_iterations}")
        iter_start_time = time.time()

        iteration_output_dir = os.path.join(molgan_output_dir, f'iteration_{iteration}')
        os.makedirs(iteration_output_dir, exist_ok=True)

        # Step 1: Train RandomForest Classifier and predict biofuel SMILES
        high_confidence_smiles, classifier_metrics = classifier_core(
            df, lotus_and_generated_smiles, confidence_threshold=confidence_threshold,
            output_dir=iteration_output_dir
        )
        classifier_performance_over_iterations.append(classifier_metrics)

        # Step 2: Filter high-confidence biofuel SMILES
        high_confidence_smiles_list = high_confidence_smiles['Smile'].tolist()

        # Step 3: Train MolGAN with high-confidence biofuel SMILES
        featurizer = create_featurizer(gan_params['num_atoms'])
        filtered_smiles = filter_smiles(high_confidence_smiles_list, gan_params['num_atoms'])
        features = featurize_molecules(featurizer, filtered_smiles)

        learning_rate = ExponentialDecay(0.001, 0.9, 5000)
        gan = create_gan_model(learning_rate, gan_params['num_atoms'])
        dataset = prepare_dataset(features)

        gan_metrics = train_gan(gan, dataset, gan_params['epochs'], gan_params['generator_steps'], gan_params['checkpoint_interval'])
        if gan_metrics is None:
            gan_metrics = {"loss": 0}  # Handle NoneType by providing default values

        molgan_performance_over_iterations.append(gan_metrics)

        generated_data = generate_data(gan, gan_params['num_samples'])
        nmols = defeaturize_generated_data(featurizer, generated_data)

        nmols = list(filter(lambda x: x is not None, nmols))
        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))

        save_generated_smiles(nmols_smiles_unique, iteration_output_dir, iteration)

        metrics = {
            "total_molecules": len(nmols),
            "unique_molecules": len(nmols_smiles_unique),
            "iteration_time": time.time() - iter_start_time
        }
        save_metrics(metrics, iteration_output_dir, iteration)

        # Step 4: Assess molecular diversity
        diversity_results, similarity_matrix, descriptors = measure_molecular_diversity(nmols_smiles_unique)
        diversity_results['iteration'] = iteration
        diversity_over_iterations.append(diversity_results)

        print(f"Diversity Results for Iteration {iteration}: {diversity_results}")

        # Save diversity plots
        plot_similarity_distribution(similarity_matrix, iteration_output_dir)
        plot_descriptor_distribution(descriptors, iteration_output_dir)

        # Step 5: Combine generated SMILES with lotus_and_generated_smiles for next iteration
        new_lotus_and_generated_smiles = pd.concat([lotus_and_generated_smiles, pd.DataFrame({'Smile': nmols_smiles_unique})])

        print(f"Iteration {iteration}/{num_iterations} completed")
        print(f"Iteration time: {metrics['iteration_time']} seconds")

        lotus_and_generated_smiles = new_lotus_and_generated_smiles.reset_index(drop=True)
        print(f"GAN Metrics: {gan_metrics}")

        print(f"Diversity Results: {diversity_results}")

    end_time = time.time()
    print(f"Total process time: {end_time - start_time:.2f} seconds")

    # Plot and save the evolution of diversity over iterations
    shannon_entropies = [res['Shannon Entropy'] for res in diversity_over_iterations]
    mean_distances = [res['Mean Distance to Centroid'] for res in diversity_over_iterations]
    iterations = list(range(1, num_iterations + 1))

    plt.figure()
    plt.plot(iterations, shannon_entropies, marker='o', label='Shannon Entropy')
    plt.plot(iterations, mean_distances, marker='o', label='Mean Distance to Centroid')
    plt.title('Evolution of Molecular Diversity Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Diversity Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(molgan_output_dir, 'diversity_evolution.png'))
    plt.close()

    # Example for plotting classifier performance (e.g., AUC)
    auc_scores = [metrics.get('AUC', None) for metrics in classifier_performance_over_iterations]

    plt.figure()
    plt.plot(iterations, auc_scores, marker='o', label='AUC Score')
    plt.title('Evolution of Classifier Performance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(molgan_output_dir, 'classifier_performance_evolution.png'))
    plt.close()

    # Example for plotting GAN performance (e.g., loss)
    gan_losses = [metrics.get('loss', None) for metrics in molgan_performance_over_iterations]

    plt.figure()
    plt.plot(iterations, gan_losses, marker='o', label='GAN Loss')
    plt.title('Evolution of MolGAN Performance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('GAN Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(molgan_output_dir, 'gan_performance_evolution.png'))
    plt.close()

if __name__ == "__main__":
    main_closed_loop()

