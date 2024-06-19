import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Features and labels
X = data.drop(columns=['Name', 'Smile', 'CAS Number', 'Type'])
y = data['Type']

# Convert labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Randomly select a portion of the data to be labeled, the rest will be unlabeled (-1)
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y_encoded)) < 0.7
y_encoded[random_unlabeled_points] = -1

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Label Spreading model
label_spread = LabelSpreading(kernel='knn', n_neighbors=7)
label_spread.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = label_spread.predict(X_test_scaled)

# Only consider the samples that are truly labeled
mask = y_test != -1
y_test_labels = le.inverse_transform(y_test[mask])
y_pred_labels = le.inverse_transform(y_pred[mask])

# Output classification report
print(classification_report(y_test_labels, y_pred_labels))

# Output accuracy
print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
