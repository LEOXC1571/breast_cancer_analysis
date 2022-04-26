# -*- coding: utf-8 -*-
# @Filename: models
# @Date: 2022-04-26 19:15
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from plot import *

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
    plot_2D_pca_analysis(feat3, target, saved)
    plot_3D_pca_analysis(feat3, target, saved)

    # output pca features
    if saved:
        pd_pca_feat = pd.DataFrame(pca_feat)
        pd_pca_feat.to_csv(os.path.join(path, 'outputs/pca_feat.csv'))
        factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
        factor_load = pd.DataFrame(factor_load)
        factor_load.to_csv(os.path.join(path, 'outputs/pca_factor_load.csv'))
    pass