import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import joblib

# Create output directory
output_dir = "SemiSupervised_output"
os.makedirs(output_dir, exist_ok=True)

# Function to read data
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df

# Function to preprocess the data
def preprocess_data(df):
    print("Preprocessing data...")
    X = df.drop(columns=['Name', 'Smile', 'CAS Number', 'Type'])
    y = df['Type']

    # Convert labels to numerical values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Randomly select a portion of the data to be labeled, the rest will be unlabeled (-1)
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_encoded)) < 0.7
    y_encoded[random_unlabeled_points] = -1

    print("Preprocessing completed.")
    return X_scaled, y_encoded, le

# Function to split data
def split_data(X, y):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test

# Function to train the Label Spreading model
def train_model(X_train, y_train):
    print("Training Label Spreading model...")
    label_spread = LabelSpreading(kernel='knn', n_neighbors=7)
    label_spread.fit(X_train, y_train)
    print("Model training completed.")
    return label_spread

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, le):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Only consider the samples that are truly labeled
    mask = y_test != -1
    y_test_labels = le.inverse_transform(y_test[mask])
    y_pred_labels = le.inverse_transform(y_pred[mask])

    # Output classification report
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    print("Classification Report:")
    print(pd.DataFrame(report).transpose())

    # Output accuracy
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print("Accuracy:", accuracy)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    return report, accuracy, conf_matrix

# Function to save results
def save_results(report, accuracy, conf_matrix, le):
    # Save classification report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"), index=True)

    # Save accuracy to a file
    with open(os.path.join(output_dir, "accuracy.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy}")

    # Save confusion matrix to CSV
    conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
    conf_matrix_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    # Plot and save confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Function to plot and save the distribution of labeled and unlabeled data
def plot_label_distribution(y, y_encoded):
    plt.figure(figsize=(10, 7))
    sns.countplot(x=y, hue=y_encoded == -1, palette='viridis')
    plt.title('Distribution of Labeled and Unlabeled Data')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend(['Labeled', 'Unlabeled'])
    plt.savefig(os.path.join(output_dir, "labeled_unlabeled_distribution.png"))
    plt.close()

# Function to save the model
def save_model(model):
    joblib.dump(model, os.path.join(output_dir, "label_spreading_model.pkl"))

# Main function
def main():
    csv_file = 'Dataset.csv'

    # Read and preprocess data
    df = read_data(csv_file)
    X, y, le = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    report, accuracy, conf_matrix = evaluate_model(model, X_test, y_test, le)

    # Save results and model
    save_results(report, accuracy, conf_matrix, le)
    save_model(model)

    # Plot and save the distribution of labeled and unlabeled data
    plot_label_distribution(df['Type'], y)

if __name__ == "__main__":
    main()
