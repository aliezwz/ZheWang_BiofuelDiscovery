import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
file_path = 'Dataset.csv'
dataset = pd.read_csv(file_path)

# 数据预处理
# 移除不必要的列
features = dataset.drop(columns=['Name', 'Smile', 'CAS Number', 'Type'])
labels = dataset['Type']

# 将标签转换为数字格式
labels = labels.map({'Biofuel': 1, 'Not biofuel': 0})

# 确保特征数据全为数值类型
features = features.apply(pd.to_numeric, errors='coerce')

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 将训练集中一部分标签设为 -1 表示无标签数据
num_unlabeled = int(0.2 * len(y_train))
indices = np.random.choice(len(y_train), num_unlabeled, replace=False)
y_train.iloc[indices] = -1

# 半监督学习 - 标签传播
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train)

# 预测
y_pred = label_prop_model.predict(X_test)

# 评估模型
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
