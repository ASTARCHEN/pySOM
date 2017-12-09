import numpy as np
import time
from pysom import SOM
from matplotlib import pyplot as plt

def circle_num(c=[0,0],r=1.0, samples_num = 1):

    t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0,samples_num,1)
    for i in i_set:
        len = np.sqrt(np.random.random()) * r
        x[i] = x[i] * len + c[0]
        y[i] = y[i] * len + c[1]
    return np.vstack((x,y)).transpose()

if __name__=="__main__":

    max_itor = 1
    t_list = np.zeros(max_itor)
    for i in range(max_itor):
        SOMNet = SOM(steps=2000,M=2, N=2)
        x1 = circle_num([5, 5], 3, 200)
        x2 = circle_num([12, 2], 1, 100)
        x3 = circle_num([10, 8], 2, 100)
        x = np.vstack((x1, x2, x3))
        dataSet = SOMNet.loadData(x)
        s = time.clock()
        C_res = SOMNet.train()
        e = time.clock()
        t_list[i] = e-s
    print("{}次运行耗时：{}".format(max_itor,t_list))
    print("平均耗时:{}".format(np.mean(t_list)))
    SOMNet.showCluster(plt)