import os
import math
import numpy as np
import torch
import torch.nn as nn

# print(np.random.randint(2, size=(10,10)))


def get_0_1_array(array, rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    # print(x)
    array = torch.from_numpy(array)
    zeros_num = int(array.shape[0] * array.shape[1] * rate)  # 根据0的比率来得到 0的个数
    new_array = np.ones(array.shape[0] * array.shape[1])  # 生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0  # 将一部分换为0
    # print(new_array)
    np.random.shuffle(new_array)  # 将0和1的顺序打乱

    re_array = new_array.reshape(array.shape)  # 重新定义矩阵的维度，与模板相同
    return re_array


def get_0_2_array(array_, rate=0.9):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(int(array_.size) * rate)  # 根据0的比率来得到 0的个数
    new_array = np.ones(array_.size)  # 生成与原来模板相同的矩阵，全为1
    print(new_array)
    new_array[:zeros_num] = 0  # 将一部分换为0
    print(new_array)
    np.random.shuffle(new_array)  # 将0和1的顺序打乱
    print(new_array)
    re_array = new_array.reshape(array_.shape)  # 重新定义矩阵的维度，与模板相同
    return re_array

if __name__ == "__main__":
    '''
    x = np.random.randint(10,size=(10,10))
    print(get_0_1_array(x))
    print(get_0_2_array(x))
    '''
    X_2dim = np.array([[1, 2, 3, 4], [2, 3, 45, 6]])
    X2_tensor = torch.from_numpy(X_2dim.astype(np.float32))
    emdeding = nn.Linear(4, 3)
    Y2 = emdeding(X2_tensor)
    print(Y2.shape)