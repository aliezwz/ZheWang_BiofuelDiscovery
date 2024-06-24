import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, \
    roc_curve, auc
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from rdkit.ML.Descriptors import MoleculeDescriptors

# Create output directory
output_dir = "GP_output"
os.makedirs(output_dir, exist_ok=True)


# Function to read data
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'Smile' in df.columns, "Smile column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df


# Function to encode the target labels
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


# Function to convert Smile to molecular descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse Smile: {smiles}"
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)


# Function to apply the descriptor function to the dataset
def apply_descriptors(df):
    print("Applying molecular descriptors...")
    df['Descriptors'] = df['Smile'].apply(smiles_to_descriptors)
    X = np.stack(df['Descriptors'])
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['Smile'].values  # Store Smile strings
    print("Descriptor conversion completed.")
    return X, y, smiles, label_encoder


# Function to split the data into training and external test sets
def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test


# Preprocessing functions
def low_variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    assert X_filtered.shape[1] > 0, "All features removed by low variance filter"
    return X_filtered


def correlation_filter(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix[column, row] > threshold for row in range(corr_matrix.shape[0]))]
    X_train_filtered = np.delete(X_train, to_drop, axis=1)
    X_test_filtered = np.delete(X_test, to_drop, axis=1)
    assert X_train_filtered.shape[1] > 0, "All features removed by correlation filter"
    return X_train_filtered, X_test_filtered


# Function to preprocess data
def preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance=True, apply_correlation_filter=True):
    print("Starting preprocessing...")
    if apply_low_variance:
        print("Applying low variance filter...")
        X_train = low_variance_filter(X_train)
        X_test = low_variance_filter(X_test)
    if apply_correlation_filter:
        print("Applying correlation filter...")
        X_train, X_test = correlation_filter(X_train, X_test)
    print("Preprocessing completed.")
    return X_train, X_test


# Function to compute all relevant metrics
def compute_metrics(y_true, y_pred, y_proba, label_encoder):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }
    return metrics


# Adjust the baseline_model function to pass the label encoder to compute_metrics
def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    print("Training baseline model...")
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    model = GaussianProcessClassifier(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Additional assertions
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print("Baseline model training completed.")
    return model, metrics


# Function to plot and save ROC curve
def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    # Find the positive label
    pos_label = label_encoder.transform(['Biofuel'])[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    print(f"Debug: FPR: {fpr}, TPR: {tpr}, Thresholds: {thresholds}")
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# Function to plot pairplot with the most important features
def plot_pairplot(df, features, target_column, filename='pairplot.png'):
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# Hyperparameter optimization using TPOT with genetic algorithm
def optimize_hyperparameters_tpot(X, y):
    print("Starting hyperparameter optimization with TPOT...")
    tpot = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization with TPOT completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()


# Hyperparameter optimization using Grid Search
def optimize_hyperparameters_grid_search(X, y):
    print("Starting hyperparameter optimization with Grid Search...")
    param_grid = {
        'kernel__k1__constant_value': [0.1, 1, 10],
        'kernel__k2__length_scale': [0.1, 1, 10],
        'kernel__k1__constant_value_bounds': [(1e-2, 1e2)],
        'kernel__k2__length_scale_bounds': [(1e-2, 1e2)],
    }
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    grid_search = GridSearchCV(estimator=GaussianProcessClassifier(kernel=kernel, random_state=42),
                               param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print("Hyperparameter optimization with Grid Search completed.")
    return grid_search.best_estimator_, grid_search.best_params_


# Function to get important features (not directly applicable for Gaussian Process, but kept for consistency)
def get_important_features(model, feature_names, top_n=5):
    return []


# Main function
def main():
    csv_file = 'Dataset.csv'
    perform_hyperparameter_optimization = True  # Boolean flag to control hyperparameter optimization
    optimization_method = 'grid_search'  # 'tpot' or 'grid_search'

    # Read and preprocess data
    df = read_data(csv_file)
    X, y, smiles, label_encoder = apply_descriptors(df)
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)
    print("Label classes:", label_encoder.classes_)

    # Preprocess the data with user options
    apply_low_variance = False
    apply_correlation_filter = False

    X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance,
                                                                apply_correlation_filter)
    assert X_train_preprocessed.shape[1] > 0 and X_test_preprocessed.shape[1] > 0, "Preprocessing removed all features"

    # Train and evaluate the baseline model with preprocessed data
    baseline_model_result, baseline_metrics = baseline_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test, label_encoder)

    print("Baseline Metrics:")
    for key, value in baseline_metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Confusion Matrix:\n{baseline_metrics['confusion_matrix']}")
    print(f"Classification Report:\n{pd.DataFrame(baseline_metrics['classification_report']).transpose()}")

    # Save baseline metrics to CSV
    baseline_metrics_df = pd.DataFrame([baseline_metrics])
    baseline_metrics_df.to_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'), index=False)

    # Plot and save ROC curve for the baseline model
    plot_and_save_roc(y_test, baseline_model_result.predict_proba(X_test_preprocessed)[:, 1], label_encoder, filename='baseline_roc_curve.png')

    if perform_hyperparameter_optimization:
        if optimization_method == 'tpot':
            # Perform hyperparameter optimization using TPOT
            best_pipeline, best_params = optimize_hyperparameters_tpot(X_train_preprocessed, y_train)
        elif optimization_method == 'grid_search':
            # Perform hyperparameter optimization using Grid Search
            best_pipeline, best_params = optimize_hyperparameters_grid_search(X_train_preprocessed, y_train)
        else:
            raise ValueError("Unknown optimization method specified.")

        # Save the best hyperparameters
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {}

    # Remove any unexpected keyword arguments from best_params
    valid_params = GaussianProcessClassifier().get_params().keys()
    best_params = {k: v for k, v in best_params.items() if k in valid_params}

    # Retrain the model with the best hyperparameters
    best_model = GaussianProcessClassifier(**best_params, random_state=42)
    best_model.fit(X_train_preprocessed, y_train)
    y_pred = best_model.predict(X_test_preprocessed)
    y_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]  # Use index 1 for the positive class

    # Additional assertions
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")

    # Evaluate the final model
    final_metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)

    # Save the final metrics and confusion matrix
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_df.to_csv(os.path.join(output_dir, 'final_model_metrics.csv'), index=False)

    confusion_matrix_df = pd.DataFrame(final_metrics['confusion_matrix'],
                                       index=['Actual Negative', 'Actual Positive'],
                                       columns=['Predicted Negative', 'Predicted Positive'])
    confusion_matrix_df.to_csv(os.path.join(output_dir, 'final_confusion_matrix.csv'))

    print("Final Model Metrics:")
    for key, value in final_metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
    print(f"Classification Report:\n{pd.DataFrame(final_metrics['classification_report']).transpose()}")

    # Plot and save ROC curve for the final model
    plot_and_save_roc(y_test, best_model.predict_proba(X_test_preprocessed)[:, 1], label_encoder, filename='final_roc_curve.png')

    # Plot pairplot with the most important features (not applicable for GP but kept for completeness)
    important_features = get_important_features(best_model, range(X_train_preprocessed.shape[1]), top_n=5)
    df_imp_features = pd.DataFrame(X_train_preprocessed, columns=range(X_train_preprocessed.shape[1]))
    df_imp_features['Type'] = y_train

    plot_pairplot(df_imp_features, important_features, 'Type', filename='important_features_pairplot.png')

    # Additional assertions for model evaluation and data integrity
    assert len(important_features) == 5, "Important features extraction did not return 5 features"
    assert len(final_metrics) > 0, "Final metrics are not computed"
    assert best_model is not None, "Best model training failed"
    assert baseline_model_result is not None, "Baseline model training failed"
    assert X_train_preprocessed.shape[0] == X_train.shape[0], "Preprocessing changed the number of training samples"
    assert X_test.shape[0] > 0, "Test set is empty after split"
    assert y_pred is not None, "Prediction failed on the test set"
    assert y_proba is not None, "Probability prediction failed on the test set"
    assert 'accuracy' in final_metrics, "Accuracy metric is missing in the final metrics"
    assert 'roc_auc' in final_metrics, "ROC AUC metric is missing in the final metrics"


if __name__ == "__main__":
    main()
