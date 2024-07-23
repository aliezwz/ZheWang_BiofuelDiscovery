import os
import time
import pandas as pd
from rdkit import Chem
from collections import OrderedDict
from deepchem.models.optimizers import ExponentialDecay
start_time = time.time()
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


def main_closed_loop():
    # Configuration
    rf_csv_file = 'C:\Users\zhe61\OneDrive\Desktop\ZHE WANG-Project 1\ZheWang_BiofuelDiscovery\Classifiers\Dataset.csv'
    lotus_csv = 'C:\Users\zhe61\OneDrive\Desktop\ZHE WANG-Project 1\ZheWang_BiofuelDiscovery\Dataset\Lotus_dataset.csv'
    molgan_output_dir = 'MolGAN_output'
    num_iterations = 2
    gan_params = {
        'num_atoms': 15,
        'epochs': 100,
        'num_samples': 10000,
        'generator_steps': 0.6,
        'checkpoint_interval': 5000,
    }

    # Read initial datasets
    df = read_data(rf_csv_file)
    lotus_and_generated_smiles = pd.read_csv(lotus_csv)

    for iteration in range(1, num_iterations + 1):
        print(f"Starting iteration {iteration}/{num_iterations}")

        # Step 1: Train RandomForest Classifier and predict biofuel SMILES
        high_confidence_smiles = classifier_core(df, lotus_and_generated_smiles, confidence_threshold=0.66)

        # Step 2: Filter high-confidence biofuel SMILES
        high_confidence_smiles_list = high_confidence_smiles['Smile'].tolist()

        # Step 3: Train MolGAN with high-confidence biofuel SMILES
        featurizer = create_featurizer(gan_params['num_atoms'])
        filtered_smiles = filter_smiles(high_confidence_smiles_list, gan_params['num_atoms'])
        features = featurize_molecules(featurizer, filtered_smiles)

        learning_rate = ExponentialDecay(0.001, 0.9, 5000)
        gan = create_gan_model(learning_rate, gan_params['num_atoms'])
        dataset = prepare_dataset(features)

        train_gan(gan, dataset, gan_params['epochs'], gan_params['generator_steps'], gan_params['checkpoint_interval'])

        generated_data = generate_data(gan, gan_params['num_samples'])
        nmols = defeaturize_generated_data(featurizer, generated_data)

        nmols = list(filter(lambda x: x is not None, nmols))
        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))

        iteration_output_dir = os.path.join(molgan_output_dir, f'iteration_{iteration}')
        save_generated_smiles(nmols_smiles_unique, iteration_output_dir, iteration)

        metrics = {
            "total_molecules": len(nmols),
            "unique_molecules": len(nmols_smiles_unique),
            "iteration_time": time.time() - start_time
        }
        save_metrics(metrics, iteration_output_dir, iteration)

        # Step 4: Combine generated SMILES with lotus_and_generated_smiles for next iteration
        new_lotus_and_generated_smiles = pd.concat([lotus_and_generated_smiles, pd.DataFrame({'Smile': nmols_smiles_unique})])

        print(f"Iteration {iteration}/{num_iterations} completed")


if __name__ == "__main__":
    main_closed_loop()
