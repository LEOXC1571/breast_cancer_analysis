# -*- coding: utf-8 -*-
# @Filename: models
# @Date: 2022-04-26 19:15
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import manifold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA

from plot import *

def save_param(model_name, dict):
    if isinstance(dict, str):
        dict = eval(dict)
    with open('outputs/' + model_name+ '_param.txt', 'w', encoding='utf-8') as f:
        f.write(str(dict))

def pca_analysis(path, data, target, saved=False):
    # perform pca analysis on breast cancer data（99.99%）
    pca = PCA(n_components=0.9999)
    pca_feat = pca.fit_transform(data)

    # explained variance ratio
    ratio = pca.explained_variance_ratio_
    print('各主成分的解释方差占比：', ratio)
    print('降维后有几个成分：', len(ratio))

    cum_ratio = np.cumsum(ratio)
    print('累计解释方差占比：', cum_ratio)

    # plot explained var ratio figure
    plot_explained_var_ratio(ratio, saved)

    # PCA选择降维保留3个主要成分
    pca_c3 = PCA(n_components=3)
    feat3 = pca_c3.fit_transform(data)

    # print(sum(pca.explained_variance_ratio_))
    # factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
    # factor_load = pd.DataFrame(factor_load)
    # factor_load.to_csv(os.path.join(current_path, 'outputs/tables/factor_load.csv'), index=None)

    # 作出样本点在二维和三维空间上的分布
    plot_pca_analysis(feat3, target, saved)

    # output pca features
    if saved:
        pd_pca_feat = pd.DataFrame(pca_feat)
        pd_pca_feat.to_csv(os.path.join(path, 'outputs/pca_feat.csv'))
        factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
        factor_load = pd.DataFrame(factor_load)
        factor_load.to_csv(os.path.join(path, 'outputs/pca_factor_load.csv'))
    pass

def kpca_analysis(path, data, target, saved=False):
    # perform pca analysis on breast cancer data（99.99%）
    clf = Pipeline([
        ('kpca', KernelPCA()),
        ('svm', svm.SVC(degree=3, gamma='auto'))
    ])
    param_grid = [{
        'kpca__n_components': [3, 4, 5, 6, 7, 8, 9, 10],
        'kpca__gamma': np.linspace(0.003, 0.03, 10),
        'kpca__kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'cosine'],
        'svm__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(data, target)
    print(grid_search.best_params_)

    best_param = grid_search.best_params_
    opt_kpca = KernelPCA(n_components=best_param['kpca__n_components'], kernel=best_param['kpca__kernel'], gamma=best_param['kpca__gamma'])
    kpca_feat = opt_kpca.fit_transform(data)

    # 作出样本点在二维和三维空间上的分布
    plot_kpca_analysis(kpca_feat, target, saved)

    # output pca features
    if saved:
        save_param('kpca', best_param)
        pd_kpca_feat = pd.DataFrame(kpca_feat)
        pd_kpca_feat.to_csv(os.path.join(path, 'outputs/kpca_feat.csv'))
    pass

def lle_analysis(path, data, target, saved=False):
    clf = Pipeline([
        ('lle', manifold.LocallyLinearEmbedding()),
        ('svm', svm.SVC(degree=3, gamma='auto'))
    ])
    param_grid = [{
        'lle__n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'lle__n_neighbors': [3, 5, 7, 10, 20, 30, 50, 100],
        'lle__method': ['standard', 'hessian', 'modified', 'ltsa'],
        'svm__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(data, target)
    print(grid_search.best_params_)

    best_param = grid_search.best_params_
    opt_lle = manifold.LocallyLinearEmbedding(n_components=best_param['lle__n_components'], n_neighbors=best_param['lle_n_neighbors'],
                                              method=best_param['lle__method'], eigen_solver='auto')
    lle_feat = opt_lle.fit_transform(data)

    if saved:
        save_param('lle', best_param)
        pd_lle_feat = pd.DataFrame(lle_feat)
        pd_lle_feat.to_csv(os.path.join(path, 'outputs/lle_feat.csv'))
    # n_neighbor_list = [2, 5, 10, 20, 50, 100]
    # for i in n_neighbor_list:
    #     LLE2 = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=i, method='standard', eigen_solver='auto').fit_transform(data)
    #     LLE3 = manifold.LocallyLinearEmbedding(n_components=3, n_neighbors=i, method='standard', eigen_solver='auto').fit_transform(data)
    #     plot_lle_analysis(LLE2, LLE3, target, i, saved)
    #
    #     if saved:
    #         pd_LLE2 = pd.DataFrame(LLE2)
    #         pd_LLE3 = pd.DataFrame(LLE3)
    #         pd_LLE2.to_csv(os.path.join(path, 'outputs/lle_' + str(i) + '_feat2d.csv'))
    #         pd_LLE3.to_csv(os.path.join(path, 'outputs/lle_' + str(i) + '_feat3d.csv'))

    pass