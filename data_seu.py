import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 设置数据目录
data_dir = './gearbox/gearset/30_2'  # 请替换为你的实际路径
file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

label_dict = {
    'Health': 0,
    'Chipped': 1,
    'Miss': 2,
    'Root': 3,
    'Surface': 4,}

X = []
y = []

for file in file_list:
    label = file.split('_')[0]  # 假设文件名如 'Chipped_20_0.csv'，label='Chipped'
    file_path = os.path.join(data_dir, file)

    # 读取数据
    df = pd.read_csv(file_path, skiprows=15, sep='\t')
    df = df.dropna(axis=1, how='all')  # 去除空列

    # 添加到数据集中
    for _, row in df.iterrows():
        X.append(row.values)
        y.append(label_dict[label])

X = np.array(X)
y = np.array(y)

# 转成 DataFrame
df = pd.DataFrame(X)
df['label'] = y  # 添加标签列

# 保存为 CSV
df.to_csv("X_y_30-2.csv", index=False)
print("X_y_30-2.csv")




