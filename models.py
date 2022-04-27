# -*- coding: utf-8 -*-
# @Filename: models
# @Date: 2022-04-26 19:15
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os

import numpy as np
import pandas as pd

from sklearn import manifold
from sklearn.decomposition import PCA, KernelPCA

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
    kpca = KernelPCA(n_components=6, kernel='rbf')
    kpca_feat = kpca.fit_transform(data)

    # PCA选择降维保留3个主要成分
    # pca_c3 = PCA(n_components=3)
    # feat3 = pca_c3.fit_transform(data)

    # print(sum(pca.explained_variance_ratio_))
    # factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
    # factor_load = pd.DataFrame(factor_load)
    # factor_load.to_csv(os.path.join(current_path, 'outputs/tables/factor_load.csv'), index=None)

    # 作出样本点在二维和三维空间上的分布
    plot_kpca_analysis(kpca_feat, target, saved)

    # output pca features
    if saved:
        pd_pca_feat = pd.DataFrame(kpca_feat)
        pd_pca_feat.to_csv(os.path.join(path, 'outputs/kpca_feat.csv'))
        # factor_load = kpca.components_.T * np.sqrt(kpca.explained_variance_)
        # factor_load = pd.DataFrame(factor_load)
        # factor_load.to_csv(os.path.join(path, 'outputs/pca_factor_load.csv'))
    pass

def lle_analysis(path, data, target, saved=False):
    n_neighbor_list = [2, 5, 10, 20, 50, 100]
    for i in n_neighbor_list:
        LLE2 = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=i, method='standard', eigen_solver='auto').fit_transform(data)
        LLE3 = manifold.LocallyLinearEmbedding(n_components=3, n_neighbors=i, method='standard', eigen_solver='auto').fit_transform(data)
        plot_lle_analysis(LLE2, LLE3, target, i, saved)

        if saved:
            pd_LLE2 = pd.DataFrame(LLE2)
            pd_LLE3 = pd.DataFrame(LLE3)
            pd_LLE2.to_csv(os.path.join(path, 'outputs/lle_' + str(i) + '_feat2d.csv'))
            pd_LLE3.to_csv(os.path.join(path, 'outputs/lle_' + str(i) + '_feat3d.csv'))

    #
    # fig = plt.figure(figsize=(15, 8))
    # fig.suptitle("Manifold Learning with %i points, %i neighbors"
    #              % (1000, n_neighbors), fontsize=14)
    #
    # # 添加三维散点图
    # ax = fig.add_subplot(251, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)
    #
    # # 创建不同的流行学习方法
    # LLE = partial(manifold.LocallyLinearEmbedding,
    #               n_neighbors, n_components, eigen_solver='auto')
    #
    # methods = OrderedDict()
    # methods['LLE'] = LLE(method='standard')
    # methods['LTSA'] = LLE(method='ltsa')
    # methods['Hessian LLE'] = LLE(method='hessian')
    # methods['Modified LLE'] = LLE(method='modified')
    # methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    # methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    # methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
    #                                            n_neighbors=n_neighbors)
    # methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
    #                                  random_state=0)
    #
    # # 展示拟合结果
    # for i, (label, method) in enumerate(methods.items()):
    #     t0 = time()
    #     Y = method.fit_transform(X)
    #     t1 = time()
    #     print("%s: %.2g sec" % (label, t1 - t0))
    #     ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    #     ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    #     ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.yaxis.set_major_formatter(NullFormatter())
    #     ax.axis('tight')
    # plt.show()
    pass