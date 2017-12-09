# -*- coding: utf-8 -*-

import numpy as np
import time
from pysom import SOM
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt



if __name__=="__main__":

    max_itor = 100
    t_list = np.zeros(max_itor)
    for i in range(max_itor):
        SOMNet = SOM()
        iris,_ = load_iris(True)
        dataSet = SOMNet.loadData(iris)
        s = time.clock()
        C_res = SOMNet.train()
        e = time.clock()
        t_list[i] = e-s
    print("{}次运行耗时：{}".format(max_itor,t_list))
    print("平均耗时:{}".format(np.mean(t_list)))
    SOMNet.showCluster(plt)