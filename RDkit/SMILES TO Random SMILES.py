import pandas as pd
from rdkit import Chem
import random


def generate_multiple_restricted_random_smiles(smiles, num_variants=3):
    """
    为给定的SMILES字符串生成多个受限随机SMILES。
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    smiles_list = []
    num_atoms = mol.GetNumAtoms()
    attempts = 0  # 计算尝试次数，避免无限循环
    while len(smiles_list) < num_variants and attempts < num_atoms * 10:
        atom_indices = list(range(num_atoms))
        random.shuffle(atom_indices)
        for start_atom_idx in atom_indices:
            if len(smiles_list) >= num_variants:
                break
            if start_atom_idx >= num_atoms:  # 在使用前验证原子索引
                continue
            try:
                random_smiles = Chem.MolToSmiles(mol, rootedAtAtom=start_atom_idx, canonical=False, doRandom=True)
                if Chem.MolFromSmiles(random_smiles) and random_smiles not in smiles_list:
                    smiles_list.append(random_smiles)
            except Exception as e:
                print(f"从索引{start_atom_idx}生成SMILES时出错: {str(e)}")
        attempts += 1
    return smiles_list


# 示例使用和错误处理
df = pd.read_csv('/content/Smiles list.csv')
smiles_column = 'SMILES'

df['Random_SMILES'] = df[smiles_column].apply(lambda x: generate_multiple_restricted_random_smiles(x, num_variants=3))
df.to_csv('RD_LIST.csv', index=False)
