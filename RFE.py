import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

data = pd.read_csv("Dataset/Dataset.csv")
print(data.head())

X = data.drop(["Name", "Smile", "CAS Number", "Type"], axis = 1)
y = data["Type"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

rf_classifier = RandomForestClassifier(n_estimators= 50, max_depth= 20, min_samples_split= 4, random_state=250)

param_grid = {
    'n_estimators' :[50, 100, 200, 1000, 1500, 2000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

rfe = RFE(rf_classifier, n_features_to_select=5, step=1)
rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]
print('Selected Features:', selected_features)

rf_classifier.fit(X_train[selected_features], y_train)

y_pred = rf_classifier.predict(X_test[selected_features])
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

feature_importances = rfe.estimator_.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(selected_features, feature_importances, color='darkblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Importance of Selected Features')
plt.xticks(rotation=45)
plt.show()