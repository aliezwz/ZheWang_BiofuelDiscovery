import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from mapie.classification import MapieClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import os

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
    print("Starting hyperparameter optimization with hyperopt...")

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
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    best_params = {
        'n_estimators': [50, 100, 200][best['n_estimators']],
        'max_features': ['sqrt', 'log2', None][best['max_features']],
        'max_depth': [None, 10, 20, 30][best['max_depth']],
        'min_samples_split': [2, 5, 10][best['min_samples_split']],
        'min_samples_leaf': [1, 2, 4][best['min_samples_leaf']],
        'bootstrap': [True, False][best['bootstrap']]
    }

    print("Hyperparameter optimization with hyperopt completed.")
    print("Best Hyperparameters:", best_params)

    best_model = RandomForestClassifier(**best_params, random_state=42)
    return best_model, best_params, trials

def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'Smile' in df.columns, "Smile column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: RDKit could not parse Smile: {smiles}")
        return [np.nan] * len(Descriptors._descList)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return descriptors

def apply_descriptors(df):
    print("Applying molecular descriptors...")
    descriptors = df['Smile'].apply(smiles_to_descriptors).apply(pd.Series)
    descriptors.columns = [desc[0] for desc in Descriptors._descList]
    df = pd.concat([df, descriptors], axis=1)
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['Smile'].values  # Store Smile strings
    df['Type'] = label_encoder.inverse_transform(y)  # Add the 'Type' column back to the dataframe
    print("Descriptor conversion completed.")
    return df, y, smiles, label_encoder

def split_data(df, y, smiles):
    print("Splitting data into training and test sets...")
    train_df, test_df, y_train, y_test, smiles_train, smiles_test = train_test_split(df, y, smiles, test_size=0.2, random_state=42)
    assert len(train_df) > 0 and len(test_df) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return train_df, test_df, y_train, y_test, smiles_train, smiles_test

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


def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    print("Training baseline model...")
    assert len(X_train) == len(y_train), f"Inconsistent train set lengths: {len(X_train)} != {len(y_train)}"
    assert len(X_test) == len(y_test), f"Inconsistent test set lengths: {len(X_test)} != {len(y_test)}"
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print label encoding information
    print(f"Label classes: {label_encoder.classes_}")
    pos_label_index = list(label_encoder.classes_).index('Biofuel')
    print(f"Index of positive class (Biofuel): {pos_label_index}")

    y_proba = model.predict_proba(X_test)[:, pos_label_index]
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print("Baseline model training completed.")
    return model, metrics

def plot_auc_curves_together(auc_data, filename='combined_auc_curves.png'):
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
    plt.savefig(os.path.join(output_dir, filename))
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

    expected_perc = np.histogram(expected_scaled, bins=breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual_scaled, bins=breakpoints)[0] / len(actual)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi

def plot_psi(train_df, test_df, external_df, feature_columns):
    psi_values = {}
    for feature in feature_columns:
        if feature in train_df.columns and feature in test_df.columns and feature in external_df.columns:
            psi_train_test = calculate_psi(train_df[feature], test_df[feature])
            psi_train_external = calculate_psi(train_df[feature], external_df[feature])
            psi_values[feature] = {'train_test': psi_train_test, 'train_external': psi_train_external}

    if not psi_values:
        print("No PSI values to plot.")
        return

    psi_df = pd.DataFrame(psi_values).transpose()
    psi_df.to_excel(os.path.join(output_dir, 'psi_values.xlsx'))

    plt.figure(figsize=(14, 8))
    sns.barplot(x=psi_df.index, y=psi_df['train_test'], color='b', label='Train-Test PSI')
    sns.barplot(x=psi_df.index, y=psi_df['train_external'], color='r', alpha=0.6, label='Train-External PSI')
    plt.xlabel('Feature')
    plt.ylabel('PSI Value')
    plt.title('Population Stability Index (PSI)')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psi_plot.png'))
    plt.close()

def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    pos_label_index = np.where(label_encoder.classes_ == 'Biofuel')[0][0]
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label_index)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 30)
    plt.ylabel('True Positive Rate', fontsize = 30)
    plt.title('Receiver Operating Characteristic', fontsize = 30)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_pairplot(df, features, target_column, filename='important_features_pairplot.png'):
    if df.empty:
        print("Cannot plot pairplot: DataFrame is empty.")
        return

    # Ensure all features are numeric
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_features:
        print("No numeric features available for pairplot.")
        return

    # Ensure target_column is in the DataFrame
    if target_column not in df.columns:
        print(f"Target column {target_column} not in DataFrame.")
        return

    sns.pairplot(df[numeric_features + [target_column]], hue=target_column)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def remove_outliers_function(train_df, y_train, smiles_train, test_df, y_test, smiles_test, feature_columns, z_threshold=3):
    from scipy import stats
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

def plot_feature_importance(model, feature_names, filename='feature_importance.png', top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]
    plt.figure()
    plt.title(f"Top {top_n} Feature Importances", fontsize = 20)
    plt.bar(range(top_n), importances[indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_precision_recall(y_true, y_proba, filename='precision_recall_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_calibration_curve(y_true, y_proba, filename='calibration_curve.png'):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def save_error_analysis(y_true, y_pred, smiles, filename='error_analysis.xlsx'):
    errors = y_true != y_pred
    error_smiles = smiles[errors]
    df_errors = pd.DataFrame({
        'Smile': error_smiles,
        'True Label': y_true[errors],
        'Predicted Label': y_pred[errors]
    })
    df_errors.to_excel(os.path.join(output_dir, filename), index=False)

def save_metrics_to_excel(metrics, filename='metrics.xlsx'):
    with pd.ExcelWriter(os.path.join(output_dir, filename)) as writer:
        pd.DataFrame(metrics['confusion_matrix']).to_excel(writer, sheet_name='Confusion Matrix')
        pd.DataFrame(metrics['classification_report']).transpose().to_excel(writer, sheet_name='Classification Report')

def save_smiles_predictions(smiles, predictions, filename_biofuel='biofuel_smiles.xlsx', filename_non_biofuel='non_biofuel_smiles.xlsx'):
    biofuel_smiles = smiles[predictions == 1]
    non_biofuel_smiles = smiles[predictions == 0]
    pd.DataFrame(biofuel_smiles, columns=['Smile']).to_excel(os.path.join(output_dir, filename_biofuel), index=False)
    pd.DataFrame(non_biofuel_smiles, columns=['Smile']).to_excel(os.path.join(output_dir, filename_non_biofuel), index=False)

def save_hyperopt_results(trials, best_params, filename='hyperopt_results.xlsx'):
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
    with pd.ExcelWriter(os.path.join(output_dir, filename)) as writer:
        df_results.to_excel(writer, sheet_name='Hyperopt Results', index=False)
        df_best_params.to_excel(writer, sheet_name='Best Parameters', index=False)

def save_acp_results(metrics, lower_bounds, upper_bounds, filename_metrics='acp_metrics.csv', filename_plot='acp_plot.png'):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, filename_metrics), index=False)

    interval_widths = upper_bounds - lower_bounds

    plt.figure()
    sns.histplot(interval_widths, kde=True)
    plt.xlabel('Prediction Interval Width')
    plt.ylabel('Frequency')
    plt.title('ACP Prediction Interval Width')
    plt.savefig(os.path.join(output_dir, filename_plot))
    plt.close()

def save_dataset_statistics(train_df, test_df, external_df, filename='dataset_statistics.xlsx'):
    with pd.ExcelWriter(os.path.join(output_dir, filename)) as writer:
        train_df.describe().to_excel(writer, sheet_name='Training Data')
        test_df.describe().to_excel(writer, sheet_name='Testing Data')
        external_df.describe().to_excel(writer, sheet_name='External Data')

def plot_boxplot(df, features, target_column, filename='boxplot.png'):
    df_melted = pd.melt(df, id_vars=[target_column], value_vars=features)
    plt.figure(figsize=(14, 8))  # Increased figure size for better readability
    sns.boxplot(x='variable', y='value', hue=target_column, data=df_melted)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Boxplot of Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def optimize_hyperparameters_tpot(X, y):
    from tpot import TPOTClassifier
    print("Starting hyperparameter optimization with TPOT...")
    tpot = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization with TPOT completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()

def optimize_hyperparameters_grid_search(X, y):
    print("Starting hyperparameter optimization with Grid Search...")
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
    print("Hyperparameter optimization with Grid Search completed.")
    return grid_search.best_estimator_, grid_search.best_params_

def get_important_features(model, feature_names, top_n=5):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n:]
    important_features = [feature_names[i] for i in indices if i < len(feature_names)]
    return important_features

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename='learning_curve.png'):
    print(f"Plotting learning curve: {title}")
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    print(f"Train sizes: {train_sizes}")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print("Training scores mean: ", train_scores_mean)
    print("Test scores mean: ", test_scores_mean)

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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Learning curve saved as {filename}")

def apply_acp(X_train, y_train, X_test, y_test, label_encoder, significance=0.1, filename='acp_metrics.csv'):
    print("Starting aggregated conformal prediction...")
    model = RandomForestClassifier(random_state=42)
    mapie = MapieClassifier(estimator=model, method="score", cv=5, n_jobs=-1)
    mapie.fit(X_train, y_train)
    prediction = mapie.predict(X_test, alpha=significance)
    y_pred = prediction[0]  # Predicted class
    lower_bounds = prediction[1][:, 0].astype(float)
    upper_bounds = prediction[1][:, 1].astype(float)

    y_proba = (lower_bounds + upper_bounds) / 2  # Estimate probabilities as mean of intervals
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    metrics['brier_score'] = brier_score_loss(y_test, y_proba)
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    metrics['ece'] = np.mean(np.abs(prob_true - prob_pred))
    metrics['mce'] = np.max(np.abs(prob_true - prob_pred))
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, filename), index=False)

    print("Aggregated conformal prediction completed.")
    return metrics, lower_bounds, upper_bounds

def print_metrics(metrics, model_name):
    print(f"{model_name} Model Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"Classification Report:\n{pd.DataFrame(metrics['classification_report']).transpose()}")

def hyperparameter_optimization(X_train, y_train, method):
    if method == 'tpot':
        return optimize_hyperparameters_tpot(X_train, y_train)
    elif method == 'grid_search':
        return optimize_hyperparameters_grid_search(X_train, y_train)
    elif method == 'hyperopt':
        return optimize_hyperparameters_hyperopt(X_train, y_train)
    else:
        raise ValueError("Unknown optimization method specified.")

def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 0]
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    return y_pred, y_proba

def low_variance_filter(train_df, test_df, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_train_filtered = selector.fit_transform(train_df.iloc[:, 4:])
    X_test_filtered = selector.transform(test_df.iloc[:, 4:])
    selected_features = train_df.columns[4:][selector.get_support(indices=True)]
    assert X_train_filtered.shape[1] > 0, "All features removed by low variance filter"
    return pd.DataFrame(X_train_filtered, columns=selected_features), pd.DataFrame(X_test_filtered, columns=selected_features), selected_features

def correlation_filter(train_df, test_df, selected_features, threshold=0.95):
    corr_matrix = train_df[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if len(to_drop) == len(selected_features):
        print("Warning: Correlation filter would remove all features. Skipping this step.")
        return train_df, test_df, selected_features

    selected_features = [feature for feature in selected_features if feature not in to_drop]
    train_df_filtered = train_df[selected_features]
    test_df_filtered = test_df[selected_features]

    if train_df_filtered.shape[1] == 0:
        print("Warning: All features removed by correlation filter. Returning original data.")
        return train_df, test_df, selected_features

    assert train_df_filtered.shape[1] > 0, "All features removed by correlation filter"
    return train_df_filtered, test_df_filtered, selected_features

def handle_infinite_and_normalize(train_df, test_df, feature_columns):
    print("Handling infinite values and normalizing data...")
    train_df[feature_columns] = train_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    print("Normalization completed.")
    return train_df, test_df, scaler

def handle_infinite_and_normalize_external(test_df, scaler, feature_columns):
    print("Handling infinite values and normalizing external data...")
    test_df[feature_columns] = test_df[feature_columns].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)

    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    # Clamp large values to a maximum threshold
    max_value = np.finfo(np.float32).max / 10  # Using a fraction of the max float32 value to avoid overflow issues
    test_df[feature_columns] = np.clip(test_df[feature_columns], -max_value, max_value)

    # Debugging: Print problematic values
    if np.any(np.isinf(test_df[feature_columns].values)):
        print("Inf values found in external data after normalization:")
        print(test_df[np.isinf(test_df[feature_columns].values)])
    if np.any(np.isnan(test_df[feature_columns].values)):
        print("NaN values found in external data after normalization:")
        print(test_df[np.isnan(test_df[feature_columns].values)])
    if np.any(test_df[feature_columns].values > max_value):
        print("Values too large found in external data after normalization and clamping:")
        print(test_df[test_df[feature_columns].values > max_value])

    print("Normalization completed for external data.")
    return test_df

def preprocess_data(train_df, y_train, smiles_train, test_df, y_test, smiles_test, apply_low_variance=True, apply_correlation_filter=True, normalize=True, remove_outliers=True, external=False, scaler=None, feature_columns=None):
    print("Starting preprocessing...")
    auc_data = {}
    if not external:
        if apply_low_variance:
            print("Applying low variance filter...")
            train_df, test_df, selected_features = low_variance_filter(train_df, test_df)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'low variance filter')
            auc_data['Low Variance Filter'] = (fpr, tpr, roc_auc)
        else:
            selected_features = train_df.columns[4:]

        if apply_correlation_filter:
            print("Applying correlation filter...")
            train_df, test_df, selected_features = correlation_filter(train_df, test_df, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'correlation filter')
            auc_data['Correlation Filter'] = (fpr, tpr, roc_auc)

        if remove_outliers:
            print("Removing outliers...")
            train_df, y_train, smiles_train, test_df, y_test, smiles_test = remove_outliers_function(train_df, y_train, smiles_train, test_df, y_test, smiles_test, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'removing outliers')
            auc_data['Removing Outliers'] = (fpr, tpr, roc_auc)

        if normalize:
            train_df, test_df, scaler = handle_infinite_and_normalize(train_df, test_df, selected_features)
            fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, selected_features, 'normalization')
            auc_data['Normalization'] = (fpr, tpr, roc_auc)
        else:
            scaler = None

        print("Selected features after preprocessing:", selected_features)
        print("Preprocessing completed.")
        return train_df, y_train, smiles_train, test_df, y_test, smiles_test, scaler, selected_features, auc_data
    else:
        assert feature_columns is not None, "feature_columns must be provided for external data preprocessing"
        test_df = handle_infinite_and_normalize_external(test_df, scaler, feature_columns)
        print("Feature columns used for external preprocessing:", feature_columns)
        print("Preprocessing completed for external data.")
        return test_df

def compute_and_print_auc(X_train, y_train, X_test, y_test, feature_columns, step_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train[feature_columns], y_train)
    y_proba = model.predict_proba(X_test[feature_columns])[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC {step_name}: {roc_auc:.4f}")
    return fpr, tpr, roc_auc

def save_external_predictions(smiles, predictions, confidences, filename_all='external_predictions_all.xlsx',
                              filename_high_conf='external_predictions_high_conf.xlsx', confidence_threshold=0.8):
    # Create a DataFrame with SMILES, predictions, and confidences
    df_all = pd.DataFrame({
        'Smile': smiles,
        'Prediction': predictions,
        'Confidence': confidences
    })

    # Save all predictions
    df_all.to_excel(os.path.join(output_dir, filename_all), index=False)

    # Filter and save high confidence predictions
    df_high_conf = df_all[df_all['Confidence'] > confidence_threshold]
    df_high_conf.to_excel(os.path.join(output_dir, filename_high_conf), index=False)

def predict_external(model, external_df, scaler, feature_columns, confidence_threshold):
    print("Starting descriptor application for external data...")
    assert 'Smile' in external_df.columns, "Smile column is missing in the external dataset"

    descriptors = external_df['Smile'].apply(smiles_to_descriptors).apply(pd.Series)
    descriptors.columns = [desc[0] for desc in Descriptors._descList]

    print(f"Generated {len(descriptors.columns)} descriptors for external data.")
    print(f"Feature columns expected: {feature_columns}")

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in descriptors.columns:
            descriptors[col] = 0  # or some default value

    descriptors = descriptors[feature_columns]

    print(f"Feature columns after selection: {descriptors.columns.tolist()}")

    external_df = pd.concat([external_df, descriptors], axis=1)

    # Preprocess external data
    external_df_preprocessed = handle_infinite_and_normalize_external(external_df, scaler, feature_columns)

    print("Making predictions on external data...")
    y_external_pred = model.predict(external_df_preprocessed[feature_columns])
    y_external_proba = model.predict_proba(external_df_preprocessed[feature_columns])[:, 1]

    save_external_predictions(external_df['Smile'], y_external_pred, y_external_proba, confidence_threshold=confidence_threshold)

    return y_external_pred, y_external_proba

def classifier_core(df, lotus_and_generated_smiles, perform_hyperparameter_optimization=False, optimization_method='grid_search',
                    plot_learning_curves=False, apply_acp_flag=False, apply_low_variance=True, apply_correlation_filter=True, normalize=True,
                    remove_outliers=False, confidence_threshold=0.8):
    print("Starting descriptor application...")
    df, y, smiles, label_encoder = apply_descriptors(df)

    print("Splitting data...")
    train_df, test_df, y_train, y_test, smiles_train, smiles_test = split_data(df, y, smiles)
    print("Label classes:", label_encoder.classes_)

    auc_data = {}

    # Print AUC before any preprocessing
    print("Evaluating model performance before any preprocessing...")
    feature_columns = train_df.columns[4:]  # Assuming features start from the 5th column onward
    fpr, tpr, roc_auc = compute_and_print_auc(train_df, y_train, test_df, y_test, feature_columns, 'before preprocessing')
    auc_data['Before Preprocessing'] = (fpr, tpr, roc_auc)

    print("Preprocessing data...")
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

    # Ensure 'Type' column is included in train_df and test_df
    train_df['Type'] = label_encoder.inverse_transform(y_train)
    test_df['Type'] = label_encoder.inverse_transform(y_test)

    print("Training and evaluating baseline model...")
    baseline_model_result, baseline_metrics = baseline_model(train_df[feature_columns], y_train, test_df[feature_columns], y_test, label_encoder)
    print_metrics(baseline_metrics, "Baseline")

    # Determine the index for the positive class (Biofuel)
    pos_label_index = list(label_encoder.classes_).index('Biofuel')

    y_proba_baseline = baseline_model_result.predict_proba(test_df[feature_columns])[:, pos_label_index]
    fpr, tpr, _ = roc_curve(y_test, y_proba_baseline)
    roc_auc = auc(fpr, tpr)
    auc_data['Baseline'] = (fpr, tpr, roc_auc)
    plot_and_save_roc(y_test, y_proba_baseline, label_encoder, filename='baseline_roc_curve.png')

    if perform_hyperparameter_optimization:
        print("Performing hyperparameter optimization...")
        best_model, best_params, trials = hyperparameter_optimization(train_df[feature_columns], y_train, optimization_method)
        save_hyperopt_results(trials, best_params)
        print("Best Hyperparameters:", best_params)
    else:
        best_model = RandomForestClassifier(bootstrap=True, max_depth=None, max_features='log2', min_samples_leaf=2, min_samples_split=10,
                                            n_estimators=200, random_state=42)

    print("Retraining the best model...")
    best_model.fit(train_df[feature_columns], y_train)
    y_pred, y_proba = make_predictions(best_model, test_df[feature_columns])
    final_metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print_metrics(final_metrics, "Final")

    y_proba_best = best_model.predict_proba(test_df[feature_columns])[:, pos_label_index]
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_auc = auc(fpr, tpr)
    auc_data['Best Model'] = (fpr, tpr, roc_auc)
    plot_and_save_roc(y_test, y_proba_best, label_encoder, filename='best_model_roc_curve.png')

    important_features = get_important_features(best_model, feature_columns, top_n=5)

    # Ensure important features are numeric and not empty
    important_numeric_features = [feature for feature in important_features if np.issubdtype(train_df[feature].dtype, np.number)]

    if important_numeric_features:
        plot_pairplot(train_df, important_numeric_features, 'Type', filename='important_features_pairplot.png')

    plot_boxplot(train_df, important_features, 'Type', filename='important_features_boxplot.png')

    save_metrics_to_excel(final_metrics, filename='final_metrics.xlsx')

    plot_feature_importance(best_model, feature_columns, filename='feature_importance.png', top_n=10)

    plot_precision_recall(y_test, y_proba_best, filename='precision_recall_curve.png')

    plot_calibration_curve(y_test, y_proba_best, filename='calibration_curve.png')

    save_error_analysis(y_test, y_pred, smiles_test, filename='error_analysis.xlsx')

    if plot_learning_curves:
        print("Plotting learning curve...")
        plot_learning_curve(best_model, 'Learning Curve', train_df[feature_columns], y_train, cv=5, filename='learning_curve.png')

    print("Predicting on external dataset...")
    external_predictions, external_proba = predict_external(best_model, lotus_and_generated_smiles, scaler, feature_columns, confidence_threshold)

    high_confidence_indices = external_proba > confidence_threshold
    high_confidence_smiles = lotus_and_generated_smiles[high_confidence_indices]

    save_smiles_predictions(lotus_and_generated_smiles['Smile'], external_predictions)

    if apply_acp_flag:
        print("Applying aggregated conformal prediction...")
        acp_metrics, lower_bounds, upper_bounds = apply_acp(train_df[feature_columns], y_train, test_df[feature_columns], y_test, label_encoder,
                                                            significance=0.1, filename='acp_metrics.csv')
        print_metrics(acp_metrics, "ACP")
        save_acp_results(acp_metrics, lower_bounds, upper_bounds)

    save_dataset_statistics(train_df, test_df, lotus_and_generated_smiles, filename='dataset_statistics.xlsx')

    plot_auc_curves_together(auc_data)

    plot_psi(train_df, test_df, lotus_and_generated_smiles, feature_columns)

    return high_confidence_smiles

def main():
    csv_file = 'Dataset_edit.csv'
    perform_hyperparameter_optimization = False
    optimization_method = 'hyperopt'
    plot_learning_curves = True
    apply_acp_flag = True
    apply_low_variance = True
    apply_correlation_filter = True
    normalize = True
    remove_outliers = False
    confidence_threshold = 0.7  # Set confidence threshold

    print("Reading dataset...")
    df = read_data(csv_file)
    print("Reading external dataset...")
    lotus_csv = "Lotus_dataset.csv"
    lotus_and_generated_smiles = pd.read_csv(lotus_csv)
    print("Starting classifier core function...")
    biofuel_smiles = classifier_core(df, lotus_and_generated_smiles, perform_hyperparameter_optimization, optimization_method,
                                     plot_learning_curves, apply_acp_flag, apply_low_variance, apply_correlation_filter, normalize, remove_outliers,
                                     confidence_threshold)
    print("Biofuel SMILES:", biofuel_smiles)

if __name__ == "__main__":
    main()

