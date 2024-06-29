import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Read data
data = pd.read_csv("Dataset for pre.csv")

# Check and handle infinite values, replace with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Prepare data
X = data.drop(["Name", "Smile", "CAS Number", "Type"], axis=1)
y = data["Type"]

# Log transform to handle extreme values
X_log_transformed = np.log1p(X)

# Fill NaN values with the mean
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X_log_transformed)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)

# Standardize data using RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and imputer
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')

# Create RandomForest classifier
rf_classifier = RandomForestClassifier(random_state=250)

# Grid search parameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Make predictions
y_pred = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output results
print('Best Parameters:', grid_search.best_params_)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_rep)
print("Classes sorted by RandomForestClassifier:", best_rf.classes_)

# Compute ROC curve data and AUC
y_scores = best_rf.predict_proba(X_test_scaled)[:, 1]  # Take the second column as the probability of the positive class
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=best_rf.classes_[1])
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
plt.savefig("ROC Curve")
plt.show()

# Feature importance
importance = best_rf.feature_importances_
imp_sort = np.argsort(importance)[::-1]
feature_name = X.columns.values

plt.figure(figsize=(10, 5))
plt.bar(feature_name[imp_sort][0:10], height=importance[imp_sort][0:10], edgecolor="k")
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Feature', fontsize=15)
plt.ylabel('Importance', fontsize=15)
plt.title('Top 10 Feature Importances', fontsize=15)
plt.savefig("Top 10")
plt.show()

# Save the model
joblib.dump(best_rf, 'best_random_forest_model.joblib')

# Load the model and scaler
model = joblib.load('best_random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
imputer = joblib.load('imputer.joblib')

# Load new dataset
new_data = pd.read_csv('LOTUS Dataset with rdkit.csv')
new_data_cleaned = new_data.drop(['SMILES', 'Name'], axis=1)
new_data_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

# Apply log transformation and fill missing values
new_data_log_transformed = np.log1p(new_data_cleaned)
new_data_filled = imputer.transform(new_data_log_transformed)

# Standardize new dataset using the same scaler
new_data_scaled = scaler.transform(new_data_filled)
new_data_scaled = new_data_scaled.astype(np.float64)  # Ensure data type is float64

# Apply the model for classification
predictions = model.predict(new_data_scaled)

# Add prediction results to the dataset
new_data['Predicted_Type'] = predictions

# Save prediction results to a new CSV file
new_data.to_csv('LOTUS_Type.csv', index=False)

print("Classification complete and saved to 'LOTUS_Type.csv'.")
