import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# 读取数据
data = pd.read_csv("Dataset for pre.csv")

# 检查并处理无限大的值，替换为NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# 准备数据
X = data.drop(["Name", "Smile", "CAS Number", "Type"], axis=1)
y = data["Type"]

# 对数变换以处理极端值
X_log_transformed = np.log1p(X)

# 使用均值填充NaN值
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X_log_transformed)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)

# 使用 RobustScaler 标准化数据
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存标准化参数和填充器
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(random_state=250)

# 网格搜索参数优化
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 获取最佳模型
best_rf = grid_search.best_estimator_

# 进行预测
y_pred = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# 输出结果
print('Best Parameters:', grid_search.best_params_)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_rep)
print("Classes sorted by RandomForestClassifier:", best_rf.classes_)

# 计算ROC曲线数据和AUC
y_scores = best_rf.predict_proba(X_test_scaled)[:, 1]  # 取第二列作为正类的概率
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=best_rf.classes_[1])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
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

# # 保存模型
# joblib.dump(best_rf, 'best_random_forest_model.joblib')
#
# # 加载模型和标准化对象
# model = joblib.load('best_random_forest_model.joblib')
# scaler = joblib.load('scaler.joblib')
# imputer = joblib.load('imputer.joblib')
#
# # 加载新数据集
# new_data = pd.read_csv('LOTUS Dataset with rdkit.csv')
# new_data_cleaned = new_data.drop(['SMILES', 'Name'], axis=1)
# new_data_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
#
# # 应用对数转换并填充缺失值
# new_data_log_transformed = np.log1p(new_data_cleaned)
# new_data_filled = imputer.transform(new_data_log_transformed)
#
# # 使用相同的标准化参数来转换新数据集
# new_data_scaled = scaler.transform(new_data_filled)
# new_data_scaled = new_data_scaled.astype(np.float64)  # 确保数据类型为 float64
#
# # 应用模型进行分类
# predictions = model.predict(new_data_scaled)
#
# # 将预测结果添加到数据集中
# new_data['Predicted_Type'] = predictions
#
# # 保存预测结果到新的CSV文件
# new_data.to_csv('LOTUS_Type.csv', index=False)
#
# print("Classification complete and saved to 'LOTUS_Type.csv'.")
