
import numpy as np
import time
from pysom import SOM
from matplotlib import pyplot as plt
if __name__=="__main__":

    max_itor = 1
    t_list = np.zeros(max_itor)
    for i in range(max_itor):
        SOMNet = SOM(steps=1500,M=3,N=3)
        dataSet = SOMNet.loadData('../../data/byq.txt', split_char=' ')
        s = time.clock()
        C_res = SOMNet.train()
        e = time.clock()
        t_list[i] = e-s
    print("{}次运行耗时：{}".format(max_itor,t_list))
    print("平均耗时:{}".format(np.mean(t_list)))
    SOMNet.showCluster(plt)
    print(SOMNet.classLabel)