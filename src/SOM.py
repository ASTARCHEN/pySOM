# -*- coding:utf-8 -*-
# 文件来源：chenge_j http://blog.csdn.net/chenge_j/article/details/72537568
# 整理：A.Star chenxiaolong12315@163.com
# 使用时请保留此信息

import numpy as np
import cmath as math
import pylab as plt
import time
import numpy as np
from scipy.linalg import norm
from scipy.spatial import distance as dist
import itertools
npa = np.array

def initCompetition(mat):
    """
    初始化输入层与竞争层神经元的连接权值矩阵
    :param mat: tuple 类型，代表输入的维度
    :return:
    """
    # 随机产生0-1之间的数作为权值
    return np.random.random(mat)

def normalize(dataSet):
    """
    对数据集进行归一化处理
    :param dataSet:
    :return:
    """
    dataSet = npa(dataSet)
    m, n = dataSet.shape
    for i in range(m):
        dataSet[i, :] = dataSet[i, :] / norm(dataSet[i, :], 2)
    return dataSet
def normalize_weight(com_weight):
    """
    对权值矩阵进行归一化处理
    :param com_weight:
    :return:
    """
    m = com_weight.shape[0]
    for ix in range(m):
        com_weight[ix] = normalize(com_weight[ix])
    return com_weight


def getWinner(data, com_weight):
    """
    得到获胜神经元的索引值
    :param data:
    :param com_weight:
    :return:
    """
    n, m, d = np.shape(com_weight)
    mat = np.zeros((n, m))
    for i in range(d):
        mat += data[i] * com_weight[:, :, i]
    arg = np.argmax(mat)
    return arg // m, arg % m

def getNeibor(n, m, N_neibor, com_weight):
    """
    得到神经元的N邻域
    :param n:
    :param m:
    :param N_neibor:
    :param com_weight:
    :return:
    """
    nn,mm, _ = np.shape(com_weight)
    i_n = range(int(max([0, n - N_neibor])), int(min([nn, n + N_neibor])),1)
    i_m = range(int(max([0, m - N_neibor])), int(min([mm, m + N_neibor])), 1)
    candidates = list(itertools.product(i_n, i_m))

    if candidates:
        N = (dist.cdist([[n, m]], candidates))[0]
        res = [(candidates[ind][0], candidates[ind][1], int(N[ind])) for ind in range(len(N)) if N[ind] <= N_neibor]
    else:
        res = []
    return res

def eta(t,N):
    """
    学习率函数
    :param t:
    :param N:
    :return:
    """
    return (0.3/(t+1)) * (np.exp(-N))
#SOM算法的实现
def do_som(dataSet, com_weight, T, N_neibor):
    """
    :param dataSet:
    :param com_weight:
    :param T: 最大迭代次数
    :param N_neibor: 初始近邻数
    :return:
    """
    for t in range(T-1):
        com_weight = normalize_weight(com_weight)
        for data in dataSet:
            n, m = getWinner(data, com_weight)
            neibor = getNeibor(n, m, N_neibor, com_weight)
            for x in neibor:
                j_n, j_m, N = x
                #权值调整
                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t,N)*(data - com_weight[j_n][j_m])
            N_neibor = N_neibor+1-(t+1)/200
    res = {}
    N, M, _ = com_weight.shape
    for i in range(len(dataSet)):
        n, m = getWinner(dataSet[i], com_weight)
        key = n*M + m
        if key in res.keys():
            res[key].append(i)
        else:
            res[key] = []
            res[key].append(i)
    return res

def SOM(dataSet,com_n,com_m,T,N_neibor):
    """
    SOM算法主方法
    :param dataSet:
    :param com_n:
    :param com_m:
    :param T:
    :param N_neibor:
    :return:
    """
    mdataSet = np.array(dataSet)
    mdataSet = normalize(dataSet=mdataSet)
    com_weight = initCompetition((com_n,com_m,np.shape(mdataSet)[1]))
    C_res = do_som(mdataSet, com_weight, T, N_neibor)
    return C_res

def draw(C , dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm', 'd']
    count = 0
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[count % len(color)], label=i)
        count += 1
    plt.legend(loc='upper right')
    plt.show()

def loadDataSet(fileName):
    arr = np.loadtxt(fileName)
    return arr.tolist()

if __name__ == '__main__':
    dataSet = loadDataSet("../data/data.txt")
    max_itor = 1
    t_list = np.zeros(max_itor)
    for i1 in range(max_itor):
        s = time.clock()
        C_res = SOM(dataSet,2,2,4,2)
        draw(C_res, dataSet)
        e = time.clock()
        t_list[i1] = e-s
    print("{}次运行耗时：{}".format(max_itor,t_list))
    print("平均耗时:{}".format(np.mean(t_list)))