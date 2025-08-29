import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

def flip_labels(labels, flip_ratio):
    """
    随机翻转输入标签中的一定比例，并返回翻转后的标签及索引。
    """
    if not (0 <= flip_ratio <= 1):
        raise ValueError("flip_ratio 必须在 [0, 1] 范围内")
    # 确定翻转的数量
    n = len(labels)
    num_to_flip = int(n * flip_ratio)
    # 随机选择需要翻转的索引
    flipped_indices = np.random.choice(n, num_to_flip, replace=False)
    # 复制原标签，防止修改原始数据
    flipped_labels = labels.copy()
    # 翻转标签
    flipped_labels[flipped_indices] = 1 - flipped_labels[flipped_indices]
    return flipped_labels, flipped_indices


# no repeat slice
def return_sequence(win_size, data):
    num = len(data)//win_size
    seq_data = np.zeros((num, win_size, data.shape[-1]))

    for i in range(num):
        seq_data[i] = data[i*win_size:(i+1)*win_size]
    return seq_data


def return_sequence_label(win_size, label):
    num = len(label)//win_size
    seq_data = np.zeros((num))

    for i in range(num):
        seq_data[i] = 1 if 1 in label[i*win_size:(i+1)*win_size] else 0
    return seq_data


def expand_data_np(data, win_size=10):
    x_data = np.zeros((data.shape[0] - win_size + 1, win_size, data.shape[-1]))
    for i in range(len(x_data)):
        x_data[i] = data[i:(i + win_size), :]
    # print(x_data.shape)
    return x_data


def cosine_similarity_torch(a, b):
    # Step 1: 归一化 a 和 b
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    
    # Step 2: 计算余弦相似度矩阵
    similarity = torch.mm(a_norm, b_norm.T)
    return similarity


class SoftCELoss_topk(object):
    def __call__(self, outputs, targets, top_k=0.2):
        loss = -torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
        vals, idx = loss.topk(int(top_k*loss.shape[0]))
        Lx = torch.mean(loss[idx])
        return Lx


class SoftCELoss(object):
    def __call__(self, outputs, targets, weight=None):
        if weight is not None:

            Lx = -torch.sum(torch.sum(F.log_softmax(outputs,
                            dim=1) * targets, dim=1)*weight)
        else:
            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs, dim=1) * targets, dim=1))
        return Lx


from numpy.testing import assert_array_almost_equal
# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1) # 1:投一次骰子， P[i, :]：各值概率， 1:一次实验
        new_y[idx] = np.where(flipped == 1)[1]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(train_labels, noise_type, nb_classes, noise_rate=0, random_state=0):
    if noise_type == 'pair':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'sym':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

# y_true = np.random.randint(10, size=10)
# print(y_true)
# train_noisy_labels, actual_noise_rate = noisify(nb_classes=10, train_labels=y_true, noise_type='symmetric', noise_rate=0.2)
# print(train_noisy_labels)




    