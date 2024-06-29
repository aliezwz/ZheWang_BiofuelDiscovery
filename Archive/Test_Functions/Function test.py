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

def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'Smile' in df.columns, "Smile column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df

csv_file = 'Dataset.csv'
df = read_data(csv_file)
# print(df)

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

y, label_encoder = encode_labels(df['Type'].values)
print(y)
print(label_encoder)

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse Smile: {smiles}"
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)

# df['Descriptors'] = df['Smile'].apply(smiles_to_descriptors)
# print(df['Descriptors'])

def apply_descriptors(df):
    print("Applying molecular descriptors...")
    df['Descriptors'] = df['Smile'].apply(smiles_to_descriptors)
    X = np.stack(df['Descriptors'])
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['Smile'].values  # Store Smile strings
    print("Descriptor conversion completed.")
    return X, y, smiles, label_encoder

# X, y, smiles, label_encoder = apply_descriptors(df)
# print(X,y,smiles,label_encoder)

def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test

X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)
print(X_train, X_test, y_train, y_test, smiles_train, smiles_test)

def low_variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    assert X_filtered.shape[1] > 0, "All features removed by low variance filter"
    return X_filtered

# X_train = low_variance_filter(X_train)
# print(X_train)

def correlation_filter(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix[column, row] > threshold for row in range(corr_matrix.shape[0]))]
    X_train_filtered = np.delete(X_train, to_drop, axis=1)
    X_test_filtered = np.delete(X_test, to_drop, axis=1)
    assert X_train_filtered.shape[1] > 0, "All features removed by correlation filter"
    return X_train_filtered, X_test_filtered

X_train, X_test = correlation_filter(X_train, X_test)
print(X_train,X_test)

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

apply_low_variance = False
apply_correlation_filter = False

X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test,
                                                            apply_low_variance,
                                                            apply_correlation_filter)
assert X_train_preprocessed.shape[1] > 0 and X_test_preprocessed.shape[1] > 0, "Preprocessing removed all features"

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

baseline_model_result, baseline_metrics = baseline_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test, label_encoder)
print(baseline_model_result,baseline_metrics)


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

metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)