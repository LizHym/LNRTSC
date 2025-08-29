import pandas as pd
import numpy as np
from utils import*

root_path = 'data/TE/'
data_path_list = []
data_path_list_te = []
for i in range(22):
    if i < 10:
        data_path_list.append(root_path + 'd0' + str(i) + '.csv')
        data_path_list_te.append(root_path + 'd0' + str(i) + '_te.csv')
    
    else:
        data_path_list.append(root_path + 'd' + str(i) + '.csv')
        data_path_list_te.append(root_path + 'd' + str(i) + '_te.csv')

win_size = 10

train_data_list = []
test_data_list = []
train_label_list = []
test_label_list = []

for i, (train_path, test_path) in enumerate(zip(data_path_list, data_path_list_te)):
    train_data = pd.read_csv(train_path, header=None).values
    train_data_list.append(expand_data_np(train_data, win_size=win_size))
    train_label_list.append(np.full(len(train_data), i))

    test_data = pd.read_csv(test_path, header=None).values
    test_data_list.append(expand_data_np(test_data, win_size=win_size))
    test_label_list.append(np.full(len(test_data), i))


print(train_data_list[0].shape)
print(test_data_list[0].shape)

