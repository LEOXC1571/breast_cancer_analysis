# -*- coding: utf-8 -*-
# @Filename: dim_reduce
# @Date: 2022-04-27 8:10
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os

import sklearn
import joblib
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

#read data
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DIM_REDUCE_MODEL = 'LLE'
feat_data = pd.read_csv(os.path.join(CURRENT_PATH, 'outputs/pca_feat.csv'), header=0)
feat_data = np.array(feat_data)
breast_cancer_data = datasets.load_breast_cancer()
origin_data = breast_cancer_data['data']
target = breast_cancer_data['target']

# use kfold to prevent overfitting
use_data = feat_data
kf = KFold(n_splits=5, shuffle=True, random_state=2022)
for train_idx, test_idx in kf.split(use_data):
    x_train, x_test = use_data[train_idx], use_data[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    clf_linear = svm.SVC(C=0.5, kernel='linear', degree=3, gamma='auto')
    clf_linear.fit(x_train, y_train)

    # calc train accuracy
    y_train_pre = clf_linear.predict(x_train)
    train_acc = sklearn.metrics.accuracy_score(y_train_pre, y_train)
    print('Training Set Accuracy of linear kernel is: %f'%train_acc)

    # calc test accuracy
    y_test_pre = clf_linear.predict(x_test)
    test_acc = sklearn.metrics.accuracy_score(y_test_pre, y_test)
    print('Test Set Accuracy of linear kernel is: %f'%test_acc)

    # calc score
    linear_score = clf_linear.score(x_test, y_test)
    print('The score of linear kernel is: %f'%linear_score)