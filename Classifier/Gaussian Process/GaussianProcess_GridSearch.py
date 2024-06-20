import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Read the dataset
df = pd.read_csv("Dataset.csv")
print(df.head(10))

# Prepare features and labels
X = df.drop(["Name", "Smile", "CAS Number", "Type"], axis=1)
y = df["Type"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define kernel functions and parameter search space
kernel_options = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=1.5),
    DotProduct(sigma_0=1.0)
]

param_grid = {
    'kernel': kernel_options,
    'max_iter_predict': [100, 200, 300],
    'multi_class': ['one_vs_rest', 'one_vs_one']
}

# Perform hyperparameter tuning using GridSearchCV
gpc = GaussianProcessClassifier(random_state=42)
grid_search = GridSearchCV(gpc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found: ", best_params)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=best_model.classes_[1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC_Curve_GridSearch.png")
plt.show()
