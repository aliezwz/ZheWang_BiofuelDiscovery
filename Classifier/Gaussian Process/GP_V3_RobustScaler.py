import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score

# 读取数据集
df = pd.read_csv("Dataset.csv")
print(df.head(10))

# 准备特征和标签
X = df.drop(["Name", "Smile", "CAS Number", "Type"], axis=1)
y = df["Type"]

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义核函数和参数搜索空间
kernel_options = [
    RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e6)),
    Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e6)),
    DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e6))
]

param_grid = {
    'kernel': kernel_options,
    'max_iter_predict': [100, 200, 300, 500],
    'multi_class': ['one_vs_rest', 'one_vs_one']
}

# 使用GridSearchCV进行超参数优化
gpc = GaussianProcessClassifier(random_state=42)
grid_search = GridSearchCV(gpc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 获取最佳参数和最佳模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found: ", best_params)

# 在测试集上评估最佳模型
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=best_model.classes_[1])
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC_Curve_V3.png")
plt.show()
