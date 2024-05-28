import deepchem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict
import time

import deepchem as dc
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

# 计时函数
def print_time(start, end, task_name):
    print(f"{task_name} 用了 {end - start:.2f} 秒")

# 开始计时加载数据集
start_time = time.time()

# 读取SMILES数据
start_time = time.time()
df = pd.read_csv("../Dataset/Smiles list.csv")
end_time = time.time()
print_time(start_time, end_time, "读取SMILES列表")

num_atoms = 12
print(df)

# data = df[['smiles']].sample(4000, random_state=42)
data = df

# 创建特征化器
start_time = time.time()
feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])
end_time = time.time()
print_time(start_time, end_time, "创建特征化器")

smiles = data['SMILES'].values

start_time = time.time()
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]
end_time = time.time()
print_time(start_time, end_time, "过滤SMILES")

# 特征化分子
start_time = time.time()
features = feat.featurize(filtered_smiles)
end_time = time.time()
print_time(start_time, end_time, "特征化分子")

indices = [i for i, data in enumerate(features) if type(data) is GraphMatrix]
print(indices)
features = [features[i] for i in indices]

# 创建模型
start_time = time.time()
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])
end_time = time.time()
print_time(start_time, end_time, "创建模型和数据集")

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}

# 训练GAN
start_time = time.time()
gan.fit_gan(iterbatches(50), generator_steps=0.2, checkpoint_interval=5000)
end_time = time.time()
print_time(start_time, end_time, "训练GAN")

# 生成数据
start_time = time.time()
generated_data = gan.predict_gan_generator(100000)
end_time = time.time()
print_time(start_time, end_time, "生成数据")

start_time = time.time()
nmols = feat.defeaturize(generated_data)
end_time = time.time()
print_time(start_time, end_time, "反特征化生成的数据")
print("{} 个分子生成".format(len(nmols)))

nmols = list(filter(lambda x: x is not None, nmols))

print("{} 个有效分子".format(len(nmols)))

# 初始化保存 SMILES 的列表
generated_smiles_list = []

nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print("{} 个独特有效分子".format(len(nmols_viz)))

# 将生成的SMILES添加到列表中
generated_smiles_list.extend(nmols_smiles_unique)

# 打印生成的SMILES
print("生成的SMILES:")
for smiles in nmols_smiles_unique:
    print(smiles)

# 保存生成的SMILES到文件
with open('generated_smiles.txt', 'w') as f:
    for smiles in generated_smiles_list:
        f.write(f"{smiles}\n")

# 生成图像
start_time = time.time()
img = Draw.MolsToGridImage(nmols_viz[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None, returnPNG=False)
end_time = time.time()
print_time(start_time, end_time, "生成图像")

# 使用matplotlib显示图像
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
