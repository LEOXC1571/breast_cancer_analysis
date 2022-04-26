# -*- coding: utf-8 -*-
# @Filename: dim_reduce
# @Date: 2022-04-26 14:00
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder

from models import *
from plot import *


# load dataset
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
breast_cancer_data = datasets.load_breast_cancer()

# print basic information
print(breast_cancer_data['DESCR'])

data = breast_cancer_data['data']
target = breast_cancer_data['target']
feat = breast_cancer_data['feature_names']

data_std = preprocessing.StandardScaler().fit_transform(data)

# perform pca analysis on breast cancer data（99.99%）

# pca_analysis(CURRENT_PATH, data_std, target, saved=True)
kpca_analysis(CURRENT_PATH, data_std, target, saved=True)
lle_analysis(CURRENT_PATH, data_std, target, saved=True)
print(data)