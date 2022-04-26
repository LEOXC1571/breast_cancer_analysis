# -*- coding: utf-8 -*-
# @Filename: plot
# @Date: 2022-04-26 14:00
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_explained_var_ratio(ratio):
    cum_ratio = np.cumsum(ratio)
    plt.figure(figsize=(8, 6))
    X = range(1, len(ratio) + 1)
    Y = ratio
    plt.bar(X, Y, edgecolor='black')
    plt.plot(X, Y, 'r.-')
    plt.plot(X, cum_ratio, 'b.-')
    plt.ylabel('explained_variance_ratio')
    plt.xlabel('PCA')
    plt.show()
    pass
