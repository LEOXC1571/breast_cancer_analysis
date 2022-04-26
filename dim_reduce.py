# -*- coding: utf-8 -*-
# @Filename: dim_reduce
# @Date: 2022-04-26 14:00
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot import *

from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder


# load dataset
breast_cancer_data = datasets.load_breast_cancer()

# print basic information
print(breast_cancer_data['DESCR'])

data = breast_cancer_data['data']
target = breast_cancer_data['target']
feat = breast_cancer_data['feature_names']

data_std = preprocessing.StandardScaler().fit_transform(data)
pd_data = pd.DataFrame(data_std)

#对数据集进行PCA降维（信息保留为99.99%）
pca = PCA(n_components=0.9999)
pca_data = pca.fit_transform(pd_data)

#降维后，每个主要成分的解释方差占比（解释PC携带的信息多少）
ratio = pca.explained_variance_ratio_
print('各主成分的解释方差占比：',ratio)
print('降维后有几个成分：',len(ratio))

cum_ratio=np.cumsum(ratio)
print('累计解释方差占比：',cum_ratio)

#绘制PCA降维后各成分方差占比的直方图和累计方差占比折线图
plot_explained_var_ratio(ratio)


#PCA选择降维保留6个主要成分
pca_c3 = PCA(n_components=3)
features3 = pca_c3.fit_transform(data_std)
# pca_feat = pd.concat([item_feature_data, pd.DataFrame(features3)], ignore_index=False, axis=1)
# pca_feat.rename(columns={0: 'pca1'}, inplace=True)
# pca_feat.rename(columns={1: 'pca2'}, inplace=True)
# pca_feat.rename(columns={2: 'pca3'}, inplace=True)
# pca_feat[['StockCode', 'pca1', 'pca2', 'pca3']].to_csv(os.path.join(current_path, 'outputs/tables/feat_pca3.csv'),index=None)
#降维后的累计各成分方差占比和（即解释PC携带的信息多少）
# print(sum(pca.explained_variance_ratio_))
# factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
# factor_load = pd.DataFrame(factor_load)
# factor_load.to_csv(os.path.join(current_path, 'outputs/tables/factor_load.csv'),index=None)
##肘方法看k值，簇内离差平方和

plt.figure(figsize=(8, 8))
plt.scatter(features3[:, 0], features3[:, 1], c=target, cmap='plasma')
for ii in np.arange(569):
    plt.text(features3[ii,0],features3[ii,1],s=None)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means PCA 2D')
plt.show()
#绘制聚类结果后3d散点图
plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='3d')
ax.scatter(features3[:, 0], features3[:, 1], features3[:, 2], c=target, cmap='plasma')
# 视角转换，转换后更易看出簇群
ax.view_init(30, 45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('K-Means PCA 3D')
plt.show()

print(data)