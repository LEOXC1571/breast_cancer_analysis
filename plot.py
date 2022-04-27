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
    plt.figure(figsize=(6, 4))
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

def plot_pca_analysis(feat, target, saved=False):
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax1.scatter(feat[:, 0], feat[:, 1], c=target, cmap='plasma')
    for ii in np.arange(569):
        plt.text(feat[ii, 0], feat[ii, 1], s=None)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Samples in 2D space')
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(feat[:, 0], feat[:, 1], feat[:, 2], c=target, cmap='plasma')
    # 视角转换，转换后更易看出簇群
    ax2.view_init(30, 45)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('Samples in 3D space')
    if saved:
        plt.savefig('outputs/pca_samples_2&3d.png')
    plt.show()
    pass

def plot_kpca_analysis(feat, target, saved=False):
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax1.scatter(feat[:, 0], feat[:, 1], c=target, cmap='coolwarm')
    for ii in np.arange(569):
        plt.text(feat[ii, 0], feat[ii, 1], s=None)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Samples in 2D space')
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(feat[:, 0], feat[:, 1], feat[:, 2], c=target, cmap='coolwarm')
    ax2.view_init(30, 45)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('Samples in 3D space')
    if saved:
        plt.savefig('outputs/kpca_samples_2&3d.png')
    plt.show()
    pass

def plot_lle_analysis(feat2d, feat3d, target, n_neighbor, saved=False):
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax1.scatter(feat2d[:, 0], feat2d[:, 1], c=target, cmap='PRGn')
    for ii in np.arange(569):
        plt.text(feat2d[ii, 0], feat2d[ii, 1], s=None)
    ax1.set_xlabel('LLE feat1')
    ax1.set_ylabel('LLE feat2')
    ax1.set_title('Samples in 2D space, n_neighbor=' + str(n_neighbor))
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(feat3d[:, 0], feat3d[:, 1], feat3d[:, 2], c=target, cmap='PRGn')
    ax2.view_init(30, 45)
    ax2.set_xlabel('LLE feat1')
    ax2.set_ylabel('LLE feat2')
    ax2.set_zlabel('LLE feat3')
    ax2.set_title('Samples in 3D space, n_neighbor=' + str(n_neighbor))
    if saved:
        plt.savefig('outputs/lle_samples_' + str(n_neighbor) + '_2&3d.png')
    plt.show()
    pass