import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from model4 import Model
from utils import*
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from knn_sim import*
from sklearn.model_selection import train_test_split

# Hyperparameters:
torch.set_default_dtype(torch.float64)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 64
lr = 0.0005
warm_epochs = 3
epochs = 5
eta_min = 0.0001
scale_style = 'dataset'
scale = None
win_size = 50
c_in = 52
num_class = 22

d_model = 512
n_heads = 4 # 被embeddings整除
e_layers = 1
d_ff = 512

discrim_hidden = [512, 256, 128]
decode_hidden = [512, 256, 128]
dropout = 0.0
act = 'gelu'
metric_avg = 'macro'

noise_type = 'pair' # 'pair', 'sym'
noise_rate =  0.2
actual_noise_rate = noise_rate
num_neigbors = 30

beta = 0.1
topk = 0.8
sigma = 5 # int
temperature = 0.5
lmbda = 0.8
n_components = 2  # DP会自动压缩
max_iter = 300
reg_covar = 0.001


root_path = 'data/TE/'
data_path_list, data_path_list_te = [], []
for i in range(num_class):
    if i < 10:
        data_path_list.append(root_path + 'd0' + str(i) + '.csv')
        data_path_list_te.append(root_path + 'd0' + str(i) + '_te.csv')
    
    else:
        data_path_list.append(root_path + 'd' + str(i) + '.csv')
        data_path_list_te.append(root_path + 'd' + str(i) + '_te.csv')

train_data_list, test_data_list = [], []
print(len(data_path_list))


for i, (train_path, test_path) in enumerate(zip(data_path_list, data_path_list_te)):
    train_data_list.append(pd.read_csv(train_path, header=None).values)
    test_data_list.append(pd.read_csv(test_path, header=None).values[160:])

print(len(train_data_list))

c_in = train_data_list[0].shape[-1]

if scale_style == 'dataset':
    train_dataset = np.concatenate(train_data_list, axis=0)
    scale = preprocessing.StandardScaler()
    scale.fit(train_dataset)

train_x_list, test_x_list, train_y_list, test_y_list = [], [], [], []
for i in range(num_class):
    train_data = scale.transform(train_data_list[i])
    train_data = expand_data_np(train_data, win_size=win_size)
    train_x_list.append(train_data)
    train_y_list.append(np.full(len(train_data), i))

    test_data = scale.transform(test_data_list[i])
    test_data = expand_data_np(test_data, win_size=win_size)
    test_x_list.append(test_data)
    test_y_list.append(np.full(len(test_data), i))

# 清醒一点！
x_data = np.concatenate(test_x_list, axis=0)
y_data = np.concatenate(test_y_list, axis=0)

y_noisy, actual_noise_rate = noisify(y_data, noise_type=noise_type, nb_classes=num_class, noise_rate=noise_rate, random_state=0)
actual_noise_rate = actual_noise_rate
print("actual_noise_rate:", actual_noise_rate)

x_data = torch.tensor(x_data)
print(x_data.shape)
y_data = torch.tensor(y_data)
y_noisy = torch.tensor(y_noisy)

    
# 生成随机的打乱顺序
original_indices = np.arange(0, len(y_data))
rng = np.random.RandomState(42)
shuffled_indices = rng.permutation(original_indices)
# 根据随机索引打乱序列
x_data = x_data[shuffled_indices]
y_data = y_data[shuffled_indices]
y_noisy = y_noisy[shuffled_indices]


index = int(len(y_data)*0.8)
ori_train_data = (x_data[:index], y_data[:index], y_noisy[:index])
train_dataset = TensorDataset(x_data[:index], y_data[:index], y_noisy[:index])
test_dataset = TensorDataset(x_data[index:], y_data[index:], y_noisy[index:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
train_loader_no_shuffle = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)


## training, validation, testing & lnl_training 函数定义
def training(model, beta, num_class, metric_avg, device, epochs, train_loader, test_loader, optimizer, scheduler, save_path):

    for ep in tqdm(range(epochs)):
        epoch_loss = 0
        model.train()
        for minibatch in train_loader:
            dict = model.update_warm(minibatch, opt=optimizer, sch=scheduler, beta=beta)
            epoch_loss += dict['loss'] # loss value as python float
        
        train_loss = epoch_loss/len(train_loader)
        vali_loss, accuracy, precision, recall, f_score = validation(model, num_class, metric_avg, device, test_loader)
        print(f"train_loss:{train_loss}, vali_loss:{vali_loss}")
        print(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, f_score:{f_score}")

    torch.save(model.state_dict(), save_path)
    return


def validation(model, num_class, metric_avg, device, test_loader):
    model.eval()
    loss_function = SoftCELoss()
    loss_sum = 0
    y_hat_list = []
    label_list = []

    for minibatch in test_loader:
        x_batch, y_batch, noi_y = minibatch
        x_batch = x_batch.to(device)
        y_batch = y_batch.long().to(device)
        y_hat, x_hat, att_list = model.forward(x_batch)
        y_hat_t = F.log_softmax(y_hat, dim=-1)

        one_hot = torch.zeros(y_batch.shape[0], num_class).to(device).scatter_(1, y_batch.view(-1, 1), 1)
        loss = loss_function(y_hat, one_hot)
        
        loss_sum +=loss
        y_hat_list.append(y_hat_t.detach().cpu().numpy())
        label_list.append(y_batch.cpu().numpy())

    vali_loss = loss_sum/len(test_loader)
    y_hats = np.concatenate(y_hat_list, axis=0)
    y_hats = np.argmax(y_hats, axis=-1)
    labels = np.concatenate(label_list, axis=0)

    accuracy = accuracy_score(labels, y_hats)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, y_hats, average=metric_avg)
    # print("VALI_Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

    return vali_loss, accuracy, precision, recall, f_score


def testing(model, num_class, metric_avg, load_path, test_loader, device='cpu'):

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()

    y_hat_list = []
    label_list = []

    for minibatch in test_loader:
        x_batch, y_batch, noi_y = minibatch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_hat_t, x_hat, att_list = model.predict(x_batch)
        y_hat_list.append(y_hat_t.detach().cpu().numpy())
        label_list.append(y_batch.cpu().numpy())

    y_hats = np.concatenate(y_hat_list, axis=0)
    y_hats = np.argmax(y_hats, axis=-1)
    labels = np.concatenate(label_list, axis=0)

    accuracy = accuracy_score(labels, y_hats)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, y_hats, average=metric_avg)
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
    cm = confusion_matrix(labels, y_hats, labels=list(range(num_class)))
    # 绘制混淆矩阵
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_class)))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.show()
    return


def eval_train_twostage(model,train_loader_no_shuffle, n_components, max_iter, reg_covar, num_neigbors, device):
    # gmm = GaussianMixture(n_components=2, max_iter=10, tol=0.001, reg_covar=0.001, warm_start=False)
    gmm = BayesianGaussianMixture(
    n_components=n_components,  # DP会自动压缩
    weight_concentration_prior_type='dirichlet_process',
    max_iter=max_iter,
    reg_covar=reg_covar,
    init_params='kmeans',
    random_state=0)

    model.eval()

    epoch_feats = []
    epoch_labels = []
    epoch_tru_labels = []

    for minibatch in train_loader_no_shuffle:
        x_batch, y_batch, noi_y = minibatch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        noi_y = noi_y.to(device)

        feats, atts = model.encode(x_batch)
        feats = F.normalize(feats.detach(), dim=1)
        epoch_feats.append(feats)
        epoch_labels.append(noi_y)
        epoch_tru_labels.append(y_batch)

    epoch_feats = torch.cat(epoch_feats, dim=0)
    epoch_feats = epoch_feats.reshape((epoch_feats.shape[0], -1))
    epoch_labels = torch.cat(epoch_labels, dim=0)
    epoch_tru_labels = torch.cat(epoch_tru_labels, dim=0)
    # Knn similarity
    K_sim = KnnSim(epoch_feats, epoch_labels, graph=num_neigbors, mode='knn')

    dataset = TensorDataset(epoch_feats, epoch_labels, epoch_tru_labels)
    feat_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False) # 节约计算空间，加速

    epoch_losses = []
    for feats, labels, t_labels in feat_loader:
        loss = K_sim(feats, labels=(labels, t_labels))  
        epoch_losses.append(loss.detach())

    epoch_losses = torch.cat(epoch_losses, dim=0).cpu().numpy()
    input_loss = epoch_losses.reshape(-1, 1)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return torch.tensor(prob)


def training_lnl(model, epochs, ori_train_data, train_loader_no_shuffle, test_loader, batch_size, optimizer, scheduler, save_path, n_components, max_iter, reg_covar, num_neigbors, num_class, metric_avg, topk, sigma, temperature, lmbda, device):

    model.to(device)

    best_acc =0.
    for ep in tqdm(range(epochs)):
        # eval label confidence
        conF = eval_train_twostage(model, train_loader_no_shuffle, n_components, max_iter, reg_covar, num_neigbors, device)

        # label correct
        dataset = TensorDataset(ori_train_data[0], ori_train_data[2], conF) # x, noi_y, F
        train_lnl_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        epoch_loss = 0.
        model.train()
        for i, minibatch in enumerate(train_lnl_loader):
            dict = model.update_with_lc(minibatch, opt=optimizer, sch=scheduler, topk=topk, sigma=sigma, temperature=temperature, lmbda=lmbda)
            epoch_loss += dict['loss'] # loss value as python float

        train_loss = epoch_loss/len(train_lnl_loader)
        vali_loss, accuracy, precision, recall, f_score = validation(model, num_class, metric_avg, device, test_loader) # args --> in_args
        print(f"train_loss:{train_loss}, vali_loss:{vali_loss}")
        print(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, f_score:{f_score}")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
    return


# 模型定义
model = Model(win_size=win_size, c_in=c_in, num_class=num_class, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, discrim_hidden=discrim_hidden, decode_hidden=decode_hidden, dropout=dropout, activation=act, output_attention=True, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=len(train_loader)*epochs)

# 模型训练
model.to(device)

# warm up and save pre-train model 
save_path = 'model_TE/warm_noi_att_te_' + noise_type + '_' + str(noise_rate) + '.pth'
print("----------Warm up-------------")
training(model, beta, num_class, metric_avg, device, warm_epochs, train_loader, test_loader, optimizer, scheduler, save_path)

# LNL training and save fine-tuned model
save_path = 'model_TE/fine_noi_att_te_' + noise_type + '_' + str(noise_rate) + '.pth'
print("----------LNL training--------")
training_lnl(model, epochs, ori_train_data, train_loader_no_shuffle, test_loader, batch_size, optimizer, scheduler, save_path, n_components, max_iter, reg_covar, num_neigbors, num_class, metric_avg, topk, sigma, temperature, lmbda, device)

# 模型测试
model = Model(win_size=win_size, c_in=c_in, num_class=num_class, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, discrim_hidden=discrim_hidden, decode_hidden=decode_hidden, dropout=dropout, activation=act, output_attention=True, device='cpu')

load_path = 'model_TE/fine_noi_att_te_' + noise_type + '_' + str(noise_rate) + '.pth'
# load_path = 'model/warm_noi_att_te_' + args.noise_type + '_' + str(args.noise_rate) + '.pth'
print("---------Testing--------------")
testing(model, num_class, metric_avg, load_path, test_loader, device='cpu')