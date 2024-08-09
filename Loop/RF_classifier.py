import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from mapie.classification import MapieClassifier
import os
import logging
import scipy.stats as stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directory
output_dir = "RF_output"
os.makedirs(output_dir, exist_ok=True)

try:
    import openpyxl
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

def optimize_hyperparameters_hyperopt(X, y):
    logging.info("Starting hyperparameter optimization with hyperopt...")

    def objective(params):
        model = RandomForestClassifier(**params, random_state=42)
        auc = cross_val_score(model, X, y, scoring='roc_auc', cv=5).mean()
        return {'loss': -auc, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.choice('n_estimators', [50, 200, 300, 500]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
        'bootstrap': hp.choice('bootstrap', [True, False])
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)

    best_params = {
        'n_estimators': [50, 200, 300, 500][best['n_estimators']],
        'max_features': ['sqrt', 'log2', None][best['max_features']],
        'max_depth': [None, 10, 20, 30][best['max_depth']],
        'min_samples_split': [2, 5, 10][best['min_samples_split']],
        'min_samples_leaf': [1, 2, 4][best['min_samples_leaf']],
        'bootstrap': [True, False][best['bootstrap']]
    }

    logging.info("Hyperparameter optimization with hyperopt completed.")
    logging.info(f"Best Hyperparameters: {best_params}")

    best_model = RandomForestClassifier(**best_params, random_state=42)
    return best_model, best_params, trials

def read_data(csv_file):
    """Read dataset from CSV file."""
    logging.info("Reading dataset...")
    df = pd.read_csv(csv_file)
    if 'Smile' not in df.columns or 'Type' not in df.columns:
        raise ValueError("Dataset must contain 'Smile' and 'Type' columns")
    logging.info("Dataset reading completed.")
    return df

def encode_labels(y):
    """Encode labels using LabelEncoder."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def smiles_to_descriptors(smiles):
    """Convert SMILES to molecular descriptors using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"RDKit could not parse Smile: {smiles}")
        return [np.nan] * len(Descriptors._descList)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    return calc.CalcDescriptors(mol)

def apply_descriptors(df):
    """Apply molecular descriptors to DataFrame."""
    logging.info("Applying molecular descriptors...")
    descriptors = df['Smile'].apply(smiles_to_descriptors).apply(pd.Series)
    descriptors.columns = [desc[0] for desc in Descriptors._descList]
    df = pd.concat([df, descriptors], axis=1)
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['Smile'].values  # Store Smile strings
    df['Type'] = label_encoder.inverse_transform(y)  # Add the 'Type' column back to the dataframe
    logging.info("Descriptor conversion completed.")
    return df, y, smiles, label_encoder

def split_data(df, y, smiles):
    """Split data into training and test sets."""
    logging.info("Splitting data into training and test sets...")
    train_df, test_df, y_train, y_test, smiles_train, smiles_test = train_test_split(df, y, smiles, test_size=0.2, random_state=42)
    assert len(train_df) > 0 and len(test_df) > 0, "Training or testing set is empty"
    logging.info("Data splitting completed.")
    return train_df, test_df, y_train, y_test, smiles_train, smiles_test

def compute_metrics(y_true, y_pred, y_proba, label_encoder):
    """Compute performance metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),  # Ensure AUC is calculated here
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }
    return metrics

def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    """Train and evaluate a baseline RandomForest model."""
    logging.info("Training baseline model...")
    assert len(X_train) == len(y_train), f"Inconsistent train set lengths: {len(X_train)} != {len(y_train)}"
    assert len(X_test) == len(y_test), f"Inconsistent test set lengths: {len(X_test)} != {len(y_test)}"
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print label encoding information
    logging.info(f"Label classes: {label_encoder.classes_}")
    pos_label_index = list(label_encoder.classes_).index('Biofuel')
    logging.info(f"Index of positive class (Biofuel): {pos_label_index}")

    y_proba = model.predict_proba(X_test)[:, pos_label_index]
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    logging.debug(f"y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    logging.info("Baseline model training completed.")
    return model, metrics

def plot_auc_curves_together(auc_data, filename='combined_auc_curves.png'):
    """Plot and save combined AUC curves."""
    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(10, 8))
    for label, (fpr, tpr, roc_auc) in auc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{label} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(filename)
    plt.close()

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    if len(expected) == 0 or len(actual) == 0:
        return 0

    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_scaled = scale_range(expected, 0, 100)
    actual_scaled = scale_range(actual, 0, 100)

    if np.max(expected_scaled) == 0 or np.max(actual_scaled) == 0:
        return 0

    epsilon = 1e-6  # small value to prevent division by zero
    expected_perc = np.histogram(expected_scaled, bins=breakpoints)[0] / len(expected) + epsilon
    actual_perc = np.histogram(actual_scaled, bins=breakpoints)[0] / len(actual) + epsilon

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi

def plot_psi(train_df, test_df, external_descriptors, feature_columns):
    """Plot and save Population Stability Index (PSI) as continuous distributions."""
    logging.info("Calculating PSI values...")

    psi_train_test_values = []
    psi_train_external_values = []

    for feature in feature_columns:
        if feature in train_df.columns and feature in test_df.columns and feature in external_descriptors.columns:
            psi_train_test = calculate_psi(train_df[feature].values, test_df[feature].values)
            psi_train_external = calculate_psi(train_df[feature].values, external_descriptors[feature].values)

            psi_train_test_values.append(psi_train_test)
            psi_train_external_values.append(psi_train_external)

    overall_psi_train_test = np.mean(psi_train_test_values)
    overall_psi_train_external = np.mean(psi_train_external_values)

    logging.info(f"Overall PSI (Train vs Test): {overall_psi_train_test}")
    logging.info(f"Overall PSI (Train vs External): {overall_psi_train_external}")

    # Plotting the PSI distributions
    plt.figure(figsize=(14, 8))
    sns.kdeplot(psi_train_test_values, label=f'Train vs Test (Overall PSI = {overall_psi_train_test:.2f})')
    sns.kdeplot(psi_train_external_values, label=f'Train vs External (Overall PSI = {overall_psi_train_external:.2f})')
    plt.xlabel('PSI Value')
    plt.ylabel('Density')
    plt.title('PSI Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psi_values_distribution.png'))
    plt.close()

    logging.info("PSI distribution plot saved as 'psi_values_distribution.png'.")

def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    """Plot and save ROC curve."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pos_label_index = np.where(label_encoder.classes_ == 'Biofuel')[0][0]
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label_index)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(filename)
    plt.close()

def plot_pairplot(df, features, target_column, filename='important_features_pairplot.png'):
    """Plot and save pairplot of important features."""
    if df.empty:
        logging.warning("Cannot plot pairplot: DataFrame is empty.")
        return

    # Ensure the features exist in the DataFrame
    valid_features = [feature for feature in features if feature in df.columns]

    if not valid_features:
        logging.warning("None of the provided features are present in the DataFrame.")
        return

    # Ensure all features are numeric
    numeric_features = df[valid_features].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_features:
        logging.warning("No numeric features available for pairplot.")
        return

    # Ensure target_column is in the DataFrame
    if target_column not in df.columns:
        logging.warning(f"Target column {target_column} not in DataFrame.")
        return

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    sns.pairplot(df[numeric_features + [target_column]], hue=target_column)
    plt.savefig(filename)
    plt.close()

def remove_outliers_function(train_df, y_train, smiles_train, test_df, y_test, smiles_test, feature_columns, z_threshold=3):
    """Remove outliers from the dataset."""
    # Compute z-scores for train data
    z_scores = stats.zscore(train_df[feature_columns])
    abs_z_scores = np.abs(z_scores)
    filtered_entries_train = (abs_z_scores < z_threshold).all(axis=1)

    train_df_filtered = train_df[filtered_entries_train]
    y_train_filtered = y_train[filtered_entries_train]
    smiles_train_filtered = smiles_train[filtered_entries_train]

    # Compute z-scores for test data
    test_z_scores = stats.zscore(test_df[feature_columns])
    test_abs_z_scores = np.abs(test_z_scores)
    filtered_entries_test = (test_abs_z_scores < z_threshold).all(axis=1)

    test_df_filtered = test_df[filtered_entries_test]
    y_test_filtered = y_test[filtered_entries_test]
    smiles_test_filtered = smiles_test[filtered_entries_test]

    return train_df_filtered, y_train_filtered, smiles_train_filtered, test_df_filtered, y_test_filtered, smiles_test_filtered

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_columns, filename='feature_importance.png', top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Ensure top_n does not exceed the length of importances
    top_n = min(top_n, len(importances))

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [feature_columns[i] for i in indices[:top_n]], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_precision_recall(y_true, y_proba, filename='precision_recall_curve.png'):
    """Plot and save precision-recall curve."""
    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(filename)
    plt.close()

def plot_calibration_curve(y_true, y_proba, filename='calibration_curve.png'):
    """Plot and save calibration curve."""
    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_calibration_curve_with_isotonic(y_true, y_proba, filename_before='calibration_curve_before.png', filename_after='calibration_curve_after.png'):
    """Plot and save calibration curve before and after applying ."""
    ir = IsotonicRegression(out_of_bounds='clip')
    y_proba_isotonic = ir.fit_transform(y_proba, y_true)

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename_before), exist_ok=True)
    os.makedirs(os.path.dirname(filename_after), exist_ok=True)

    # Before calibration
    prob_true_before, prob_pred_before = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred_before, prob_true_before, marker='o', linewidth=1, label='Before Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve Before')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_before)
    plt.close()

    # After calibration
    prob_true_after, prob_pred_after = calibration_curve(y_true, y_proba_isotonic, n_bins=10, strategy='uniform')
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred_after, prob_true_after, marker='o', linewidth=1, label='After Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve After')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_after)
    plt.close()

def save_error_analysis(y_true, y_pred, smiles, filename='error_analysis.xlsx'):
    """Save error analysis to an Excel file."""
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    errors = y_true != y_pred
    error_smiles = smiles[errors]
    df_errors = pd.DataFrame({
        'Smile': error_smiles,
        'True Label': y_true[errors],
        'Predicted Label': y_pred[errors]
    })
    df_errors.to_excel(filename, index=False)

def save_metrics_to_excel(metrics, filename='metrics.xlsx'):
    """Save metrics to an Excel file."""
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(metrics['confusion_matrix']).to_excel(writer, sheet_name='Confusion Matrix')
        pd.DataFrame(metrics['classification_report']).transpose().to_excel(writer, sheet_name='Classification Report')

def save_smiles_predictions(smiles, predictions, filename_biofuel='biofuel_smiles.xlsx', filename_non_biofuel='non_biofuel_smiles.xlsx'):
    """Save SMILES predictions to Excel files."""
    # Ensure the directory exists before saving the files
    os.makedirs(os.path.dirname(filename_biofuel), exist_ok=True)
    os.makedirs(os.path.dirname(filename_non_biofuel), exist_ok=True)

    biofuel_smiles = smiles[predictions == 1]
    non_biofuel_smiles = smiles[predictions == 0]
    pd.DataFrame(biofuel_smiles, columns=['Smile']).to_excel(filename_biofuel, index=False)
    pd.DataFrame(non_biofuel_smiles, columns=['Smile']).to_excel(filename_non_biofuel, index=False)

def save_hyperopt_results(trials, best_params, filename='hyperopt_results.xlsx'):
    """Save hyperopt results to an Excel file."""
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    results = []
    for trial in trials.trials:
        result = trial['result']
        params = {key: value[0] if isinstance(value, list) else value for key, value in trial['misc']['vals'].items()}
        params.update(result)
        results.append(params)
    df_results = pd.DataFrame(results)
    df_results['auc'] = -df_results['loss']  # Convert loss to AUC
    df_best_params = pd.DataFrame([best_params])
    df_best_params['auc'] = df_results['auc'].max()
    with pd.ExcelWriter(filename) as writer:
        df_results.to_excel(writer, sheet_name='Hyperopt Results', index=False)
        df_best_params.to_excel(writer, sheet_name='Best Parameters', index=False)

def save_acp_results(metrics, y_pred_sets, filename_metrics='acp_metrics.csv'):
    """Save ACP results to CSV."""
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(filename_metrics), exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(filename_metrics, index=False)

def save_dataset_statistics(train_df, test_df, external_df, filename='dataset_statistics.xlsx'):
    """Save dataset statistics to an Excel file."""
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with pd.ExcelWriter(filename) as writer:
        train_df.describe().to_excel(writer, sheet_name='Training Data')
        test_df.describe().to_excel(writer, sheet_name='Testing Data')
        external_df.describe().to_excel(writer, sheet_name='External Data')

def plot_boxplot(df, features, target_column, filename='boxplot.png'):
    """Plot and save boxplot of important features."""
    df_melted = pd.melt(df, id_vars=[target_column], value_vars=features)

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(14, 8))  # Increased figure size for better readability
    sns.boxplot(x='variable', y='value', hue=target_column, data=df_melted)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Boxplot of Important Features')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def optimize_hyperparameters_tpot(X, y):
    """Optimize hyperparameters using TPOT."""
    from tpot import TPOTClassifier
    logging.info("Starting hyperparameter optimization with TPOT...")
    tpot = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    logging.info("Hyperparameter optimization with TPOT completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()

def optimize_hyperparameters_grid_search(X, y):
    """Optimize hyperparameters using Grid Search."""
    logging.info("Starting hyperparameter optimization with Grid Search...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    logging.info("Hyperparameter optimization with Grid Search completed.")
    return grid_search.best_estimator_, grid_search.best_params_

def get_important_features(model, feature_names, top_n=5):
    """Get the most important features from the model."""
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n:]
    important_features = [feature_names[i] for i in indices if i < len(feature_names)]
    return important_features

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename='learning_curve.png'):
    """Plot and save learning curve."""
    logging.info(f"Plotting learning curve: {title}")

    # Ensure the directory exists before saving the plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    logging.info(f"Train sizes: {train_sizes}")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    logging.info(f"Training scores mean: {train_scores_mean}")
    logging.info(f"Test scores mean: {test_scores_mean}")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()
    logging.info(f"Learning curve saved as {filename}")

def apply_acp(X_train, y_train, X_test, y_test, label_encoder, best_params, significance=0.1, filename='acp_metrics.csv'):
    """Apply Aggregated Conformal Prediction (ACP) using best model parameters."""
    logging.info("Starting aggregated conformal prediction...")

    # Ensure random_state is not included in best_params to avoid duplication
    if 'random_state' in best_params:
        del best_params['random_state']

    model = RandomForestClassifier(**best_params, random_state=42)
    mapie = MapieClassifier(estimator=model, method="score", cv=5, n_jobs=-1)
    mapie.fit(X_train, y_train)
    prediction = mapie.predict(X_test, alpha=significance)
    y_pred = prediction[0]  # Predicted class
    y_pred_sets = prediction[1].astype(int)

    metrics = compute_metrics(y_test, y_pred, y_pred_sets[:, 1], label_encoder)
    metrics['brier_score'] = brier_score_loss(y_test, y_pred_sets[:, 1])

    try:
        prob_true, prob_pred = calibration_curve(y_test, y_pred_sets[:, 1], n_bins=10, strategy='uniform')
        metrics['ece'] = np.mean(np.abs(prob_true - prob_pred))
        metrics['mce'] = np.max(np.abs(prob_true - prob_pred))
    except ImportError:
        logging.error("calibration_curve is not available in this version of scikit-learn. Please update scikit-learn to use this function.")
        metrics['ece'] = 'N/A'
        metrics['mce'] = 'N/A'

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, filename), index=False)

    logging.info("Aggregated conformal prediction completed.")
    return metrics, y_pred_sets

def print_metrics(metrics, model_name):
    """Print model metrics."""
    logging.info(f"{model_name} Model Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            logging.info(f"{key}: {value}")
    logging.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    logging.info(f"Classification Report:\n{pd.DataFrame(metrics['classification_report']).transpose()}")

def hyperparameter_optimization(X_train, y_train, method):
    """Optimize hyperparameters using specified method."""
    if method == 'tpot':
        return optimize_hyperparameters_tpot(X_train, y_train)
    elif method == 'grid_search':
        return optimize_hyperparameters_grid_search(X_train, y_train)
    elif method == 'hyperopt':
        return optimize_hyperparameters_hyperopt(X_train, y_train)
    else:
        raise ValueError("Unknown optimization method specified.")

def make_predictions(model, X_test):
    """Make predictions and compute probabilities."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    return y_pred, y_proba

def low_variance_filter(train_df, test_df, threshold=0.01):
    """Filter features with low variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_train_filtered = selector.fit_transform(train_df.iloc[:, 4:])
    X_test_filtered = selector.transform(test_df.iloc[:, 4:])
    selected_features = train_df.columns[4:][selector.get_support(indices=True)]
    assert X_train_filtered.shape[1] > 0, "All features removed by low variance filter"
    return pd.DataFrame(X_train_filtered, columns=selected_features), pd.DataFrame(X_test_filtered, columns=selected_features), selected_features

def correlation_filter(train_df, test_df, selected_features, threshold=0.95):
    """Filter features with high correlation."""
    corr_matrix = train_df[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if len(to_drop) == len(selected_features):
        logging.warning("Correlation filter would remove all features. Skipping this step.")
        return train_df, test_df, selected_features

    selected_features = [feature for feature in selected_features if feature not in to_drop]
    train_df_filtered = train_df[selected_features]
    test_df_filtered = test_df[selected_features]

    if train_df_filtered.shape[1] == 0:
        logging.warning("All features removed by correlation filter. Returning original data.")
        return train_df, test_df, selected_features

    assert train_df_filtered.shape[1] > 0, "All features removed by correlation filter"
    return train_df_filtered, test_df_filtered, selected_features

def handle_infinite_and_normalize(train_df, test_df, feature_columns):
    """Handle infinite values and normalize data."""
    logging.info("Handling infinite values and normalizing data...")
    train_df[feature_columns] = train_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    logging.info("Normalization completed.")
    return train_df, test_df, scaler

def handle_infinite_and_normalize_external(test_df, scaler, feature_columns):
    """Handle infinite values and normalize external data."""
    logging.info("Handling infinite values and normalizing external data...")
    test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    # Clamp large values to a maximum threshold
    max_value = np.finfo(np.float32).max / 10  # Using a fraction of the max float32 value to avoid overflow issues
    test_df[feature_columns] = np.clip(test_df[feature_columns], -max_value, max_value)

    # Debugging: Print problematic values
    if np.any(np.isinf(test_df[feature_columns].values)):
        logging.warning("Inf values found in external data after normalization:")
        logging.warning(test_df[np.isinf(test_df[feature_columns].values)])
    if np.any(np.isnan(test_df[feature_columns].values)):
        logging.warning("NaN values found in external data after normalization:")
        logging.warning(test_df[np.isnan(test_df[feature_columns].values)])
    if np.any(test_df[feature_columns].values > max_value):
        logging.warning("Values too large found in external data after normalization and clamping:")
        logging.warning(test_df[test_df[feature_columns].values > max_value])

    logging.info("Normalization completed for external data.")
    return test_df

def preprocess_data(train_df, y_train, smiles_train, test_df, y_test, smiles_test, apply_low_variance=True, apply_correlation_filter=True, normalize=True, remove_outliers=True, external=False, scaler=None, feature_columns=None):
    """Preprocess data with various filters and normalization."""
    logging.info("Starting preprocessing...")
    auc_data = {}
    if not external:
        if apply_low_variance:
            logging.info("Applying low variance filter...")
            train_df, test_df, selected_features = low_variance_filter(train_df, test_df)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'low variance filter')
            auc_data['Low Variance Filter'] = (fpr, tpr, roc_auc)
        else:
            selected_features = train_df.columns[4:]

        if apply_correlation_filter:
            logging.info("Applying correlation filter...")
            train_df, test_df, selected_features = correlation_filter(train_df, test_df, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'correlation filter')
            auc_data['Correlation Filter'] = (fpr, tpr, roc_auc)

        if remove_outliers:
            logging.info("Removing outliers...")
            train_df, y_train, smiles_train, test_df, y_test, smiles_test = remove_outliers_function(train_df, y_train, smiles_train, test_df, y_test, smiles_test, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'removing outliers')
            auc_data['Removing Outliers'] = (fpr, tpr, roc_auc)

        if normalize:
            train_df, test_df, scaler = handle_infinite_and_normalize(train_df, test_df, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'normalization')
            auc_data['Normalization'] = (fpr, tpr, roc_auc)
        else:
            scaler = None

        logging.info(f"Selected features after preprocessing: {selected_features}")
        logging.info("Preprocessing completed.")
        return train_df, y_train, smiles_train, test_df, y_test, smiles_test, scaler, selected_features, auc_data
    else:
        assert feature_columns is not None, "feature_columns must be provided for external data preprocessing"
        test_df = handle_infinite_and_normalize_external(test_df, scaler, feature_columns)
        logging.info(f"Feature columns used for external preprocessing: {feature_columns}")
        logging.info("Preprocessing completed for external data.")
        return test_df

def compute_and_print_auc(X_train, y_train, X_test, y_test, feature_columns, step_name):
    """Compute and print AUC score."""
    assert len(X_train) > 0, "Training set is empty."
    assert len(X_test) > 0, "Test set is empty."
    assert len(feature_columns) > 0, "Feature columns are empty."

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train[feature_columns], y_train)
    y_proba = model.predict_proba(X_test[feature_columns])[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    logging.info(f"AUC {step_name}: {roc_auc:.4f}")
    return fpr, tpr, roc_auc

def save_external_predictions(smiles, predictions, confidences, filename_all='external_predictions_all.xlsx',
                              filename_high_conf='external_predictions_high_conf.xlsx', confidence_threshold=0.8):
    """Save external predictions to Excel files."""
    df_all = pd.DataFrame({
        'Smile': smiles,
        'Prediction': predictions,
        'Confidence': confidences
    })

    df_all.to_excel(os.path.join(output_dir, filename_all), index=False)

    df_high_conf = df_all[df_all['Confidence'] > confidence_threshold]
    df_high_conf.to_excel(os.path.join(output_dir, filename_high_conf), index=False)

def predict_external(model, external_df, scaler, feature_columns, confidence_threshold):
    """Make predictions on external dataset using the best model with isotonic regression calibration."""
    logging.info("Starting descriptor application for external data...")
    assert 'Smile' in external_df.columns, "Smile column is missing in the external dataset"

    descriptors = external_df['Smile'].apply(smiles_to_descriptors).apply(pd.Series)
    descriptors.columns = [desc[0] for desc in Descriptors._descList]

    logging.info(f"Generated {len(descriptors.columns)} descriptors for external data.")
    logging.info(f"Feature columns expected: {feature_columns}")

    for col in feature_columns:
        if col not in descriptors.columns:
            descriptors[col] = 0  # or some default value

    descriptors = descriptors[feature_columns]

    logging.info(f"Feature columns after selection: {descriptors.columns.tolist()}")

    external_df = pd.concat([external_df, descriptors], axis=1)

    external_df_preprocessed = handle_infinite_and_normalize_external(external_df, scaler, feature_columns)

    logging.info("Applying isotonic regression calibration...")
    ir = IsotonicRegression(out_of_bounds='clip')
    y_proba_train = model.predict_proba(external_df_preprocessed[feature_columns])[:, 1]  # Probabilities from the model
    y_proba_isotonic = ir.fit_transform(y_proba_train, y_proba_train)  # Fit and transform using isotonic regression

    logging.info("Making predictions on external data...")
    y_external_pred = (y_proba_isotonic > confidence_threshold).astype(int)  # Binarize predictions based on confidence threshold
    y_external_proba = y_proba_isotonic

    save_external_predictions(external_df['Smile'], y_external_pred, y_external_proba, confidence_threshold=confidence_threshold)

    return y_external_pred, y_external_proba, descriptors

def classifier_core(df, lotus_and_generated_smiles, perform_hyperparameter_optimization=False, optimization_method='grid_search',
                    plot_learning_curves=False, apply_acp_flag=False, apply_low_variance=True, apply_correlation_filter=True, normalize=True,
                    remove_outliers=False, confidence_threshold=0.8, user_defined_features=None, output_dir=None):
    logging.info("Starting descriptor application...")
    df, y, smiles, label_encoder = apply_descriptors(df)

    logging.info("Splitting data...")
    train_df, test_df, y_train, y_test, smiles_train, smiles_test = split_data(df, y, smiles)
    logging.info(f"Label classes: {label_encoder.classes_}")

    auc_data = {}

    # Print AUC before any preprocessing
    logging.info("Evaluating model performance before any preprocessing...")
    feature_columns = train_df.columns[4:]  # Assuming features start from the 5th column onward
    fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, feature_columns, 'before preprocessing')
    auc_data['Before Preprocessing'] = (fpr, tpr, roc_auc)

    logging.info("Training and evaluating baseline model...")
    baseline_model_result, baseline_metrics = baseline_model(train_df[feature_columns], y_train, test_df[feature_columns], y_test, label_encoder)
    print_metrics(baseline_metrics, "Baseline")

    pos_label_index = list(label_encoder.classes_).index('Biofuel')

    y_proba_baseline = baseline_model_result.predict_proba(test_df[feature_columns])[:, pos_label_index]
    fpr, tpr, _ = roc_curve(y_test, y_proba_baseline)
    roc_auc = auc(fpr, tpr)
    auc_data['Baseline'] = (fpr, tpr, roc_auc)
    plot_and_save_roc(y_test, y_proba_baseline, label_encoder, filename=os.path.join(output_dir, 'baseline_roc_curve.png'))

    logging.info("Preprocessing data...")
    train_df, y_train, smiles_train, test_df, y_test, smiles_test, scaler, feature_columns, preprocessing_auc_data = preprocess_data(train_df,
                                                                                                                                     y_train,
                                                                                                                                     smiles_train,
                                                                                                                                     test_df, y_test,
                                                                                                                                     smiles_test,
                                                                                                                                     apply_low_variance,
                                                                                                                                     apply_correlation_filter,
                                                                                                                                     normalize,
                                                                                                                                     remove_outliers)
    auc_data.update(preprocessing_auc_data)

    assert train_df.shape[1] > 0 and test_df.shape[1] > 0, "Preprocessing removed all features"

    train_df['Type'] = label_encoder.inverse_transform(y_train)
    test_df['Type'] = label_encoder.inverse_transform(y_test)

    if perform_hyperparameter_optimization:
        logging.info("Performing hyperparameter optimization...")
        best_model, best_params, trials = hyperparameter_optimization(train_df[feature_columns], y_train, optimization_method)
        save_hyperopt_results(trials, best_params, filename=os.path.join(output_dir, 'hyperopt_results.xlsx'))
        logging.info(f"Best Hyperparameters: {best_params}")
    else:
        best_params = {
            'bootstrap': True,
            'max_depth': None,
            'max_features': 'log2',
            'min_samples_leaf': 4,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 42
        }
        best_model = RandomForestClassifier(**best_params)

    logging.info("Retraining the best model...")
    best_model.fit(train_df[feature_columns], y_train)
    y_pred, y_proba = make_predictions(best_model, test_df[feature_columns])
    final_metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print_metrics(final_metrics, "Final")

    y_proba_best = best_model.predict_proba(test_df[feature_columns])[:, pos_label_index]
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_auc = auc(fpr, tpr)
    auc_data['Best Model'] = (fpr, tpr, roc_auc)
    plot_and_save_roc(y_test, y_proba_best, label_encoder, filename=os.path.join(output_dir, 'best_model_roc_curve.png'))

    important_features = get_important_features(best_model, feature_columns, top_n=5)

    important_numeric_features = [feature for feature in important_features if np.issubdtype(train_df[feature].dtype, np.number)]

    if important_numeric_features:
        plot_pairplot(train_df, important_numeric_features, 'Type', filename=os.path.join(output_dir, 'important_features_pairplot.png'))

    plot_boxplot(train_df, important_features, 'Type', filename=os.path.join(output_dir, 'important_features_boxplot.png'))

    save_metrics_to_excel(final_metrics, filename=os.path.join(output_dir, 'final_metrics.xlsx'))

    plot_feature_importance(best_model, feature_columns, filename=os.path.join(output_dir, 'feature_importance.png'), top_n=10)

    plot_precision_recall(y_test, y_proba_best, filename=os.path.join(output_dir, 'precision_recall_curve.png'))

    plot_calibration_curve(y_test, y_proba_best, filename=os.path.join(output_dir, 'calibration_curve.png'))
    plot_calibration_curve_with_isotonic(y_test, y_proba_best, filename_before=os.path.join(output_dir, 'calibration_curve_before.png'), filename_after=os.path.join(output_dir, 'calibration_curve_after.png'))

    save_error_analysis(y_test, y_pred, smiles_test, filename=os.path.join(output_dir, 'error_analysis.xlsx'))

    if plot_learning_curves:
        logging.info("Plotting learning curve...")
        plot_learning_curve(best_model, 'Learning Curve', train_df[feature_columns], y_train, cv=5, filename=os.path.join(output_dir, 'learning_curve.png'))

    if apply_acp_flag:
        logging.info("Applying aggregated conformal prediction...")
        acp_metrics, y_pred_sets = apply_acp(train_df[feature_columns], y_train, test_df[feature_columns], y_test, label_encoder, best_params,
                                             significance=0.1, filename=os.path.join(output_dir, 'acp_metrics.csv'))
        print_metrics(acp_metrics, "ACP")
        save_acp_results(acp_metrics, y_pred_sets, filename_metrics=os.path.join(output_dir, 'acp_metrics.csv'))

    logging.info("Predicting on external dataset...")
    external_predictions, external_proba, external_descriptors = predict_external(best_model, lotus_and_generated_smiles, scaler, feature_columns,
                                                                                  confidence_threshold)

    high_confidence_indices = external_proba > confidence_threshold
    high_confidence_smiles = lotus_and_generated_smiles[high_confidence_indices]

    save_smiles_predictions(lotus_and_generated_smiles['Smile'], external_predictions, filename_biofuel=os.path.join(output_dir, 'biofuel_smiles.xlsx'), filename_non_biofuel=os.path.join(output_dir, 'non_biofuel_smiles.xlsx'))

    save_dataset_statistics(train_df, test_df, lotus_and_generated_smiles, filename=os.path.join(output_dir, 'dataset_statistics.xlsx'))

    plot_auc_curves_together(auc_data, filename=os.path.join(output_dir, 'combined_auc_curves.png'))

    # Calculate and plot PSI values using the precomputed external descriptors
    logging.info("Calculating and plotting PSI values...")
    plot_psi(train_df, test_df, external_descriptors, feature_columns)

    if user_defined_features:
        plot_pairplot(train_df, user_defined_features, 'Type', filename=os.path.join(output_dir, 'user_defined_features_pairplot.png'))

    return high_confidence_smiles, final_metrics

def main():
    csv_file = 'Seed Dataset.csv'
    perform_hyperparameter_optimization = False
    optimization_method = 'hyperopt'
    plot_learning_curves = True
    apply_acp_flag = True
    apply_low_variance = True
    apply_correlation_filter = True
    normalize = True
    remove_outliers = False
    confidence_threshold = 0.6  # Set confidence threshold
    user_defined_features = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names

    logging.info("Reading dataset...")
    df = read_data(csv_file)
    logging.info("Reading external dataset...")
    lotus_csv = "Lotus_dataset.csv"
    lotus_and_generated_smiles = pd.read_csv(lotus_csv)
    logging.info("Starting classifier core function...")
    biofuel_smiles = classifier_core(df, lotus_and_generated_smiles, perform_hyperparameter_optimization, optimization_method,
                                     plot_learning_curves, apply_acp_flag, apply_low_variance, apply_correlation_filter, normalize, remove_outliers,
                                     confidence_threshold, user_defined_features)
    logging.info(f"Biofuel SMILES: {biofuel_smiles}")

if __name__ == "__main__":
    main()
