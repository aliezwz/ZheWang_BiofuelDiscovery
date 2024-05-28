import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score

df = pd.read_csv("Dataset.csv")
print(df.head(10))

X = df.drop(["Name", "Smile", "CAS Number", "Type"], axis=1)
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernel =1.0 * RBF(length_scale= 1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state= 42)

gpc.fit(X_train,y_train)

y_pred = gpc.predict(X_test)
y_proba = gpc.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

print("Classes sorted by RandomForestClassifier:", gpc.classes_)

y_scores = gpc.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label= gpc.classes_[1])
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
plt.savefig("ROC Curve_V1")
plt.show()
