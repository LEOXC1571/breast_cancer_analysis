# -*- coding: utf-8 -*-
# @Filename: plot
# @Date: 2022-04-26 14:00
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_explained_var_ratio(ratio, saved=False):
    cum_ratio = np.cumsum(ratio)
    plt.figure(figsize=(8, 6))
    X = range(1, len(ratio) + 1)
    Y = ratio
    plt.bar(X, Y, edgecolor='black')
    plt.plot(X, Y, 'r.-')
    plt.plot(X, cum_ratio, 'b.-')
    plt.ylabel('explained_variance_ratio')
    plt.xlabel('PCA')
    if saved:
        plt.savefig('outputs/pca_explained_var_ratio.png')
    plt.show()
    pass

def plot_2D_pca_analysis(feat2d, target, saved=False):
    plt.figure(figsize=(8, 8))
    plt.scatter(feat2d[:, 0], feat2d[:, 1], c=target, cmap='plasma')
    for ii in np.arange(569):
        plt.text(feat2d[ii, 0], feat2d[ii, 1], s=None)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Samples in 2D space')
    if saved:
        plt.savefig('outputs/pca_samples_2d.png')
    plt.show()
    pass

def plot_3D_pca_analysis(feat3d, target, saved=False):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(feat3d[:, 0], feat3d[:, 1], feat3d[:, 2], c=target, cmap='plasma')
    # 视角转换，转换后更易看出簇群
    ax.view_init(30, 45)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Samples in 3D space')
    if saved:
        plt.savefig('outputs/pca_samples_3d.png')
    plt.show()
    pass