import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
data = pd.read_csv("Dataset.csv")

# 特征和标签
X = data.drop(columns=['Name', 'Smile', 'CAS Number', 'Type'])
y = data['Type']

# 将标签转换为数字
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 随机选择一部分数据进行标记，其余数据设为未标记 (-1)
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y_encoded)) < 0.7
y_encoded[random_unlabeled_points] = -1

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 创建并训练Label Spreading模型
label_spread = LabelSpreading(kernel='knn', n_neighbors=7)
label_spread.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = label_spread.predict(X_test)

# 只考虑那些真实标记的数据
mask = y_test != -1
y_test_labels = le.inverse_transform(y_test[mask])
y_pred_labels = le.inverse_transform(y_pred[mask])

# 输出分类报告
print(classification_report(y_test_labels, y_pred_labels))

# 输出准确率
print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
