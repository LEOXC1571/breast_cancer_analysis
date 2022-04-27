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

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
feat_data = pd.read_csv(os.path.join(CURRENT_PATH, 'outputs/lle_10_feat2d.csv'))
breast_cancer_data = datasets.load_breast_cancer()

# print basic information
print(breast_cancer_data['DESCR'])

data = breast_cancer_data['data']
target = breast_cancer_data['target']
feat = breast_cancer_data['feature_names']


decision_train = []
predict_train = []
readdata = np.load('./dataset/dataset.npy')
testdata = np.loadtxt('./dataset/test.txt', delimiter=':')
# print(testdata)
x, y = np.split(readdata, (4,), axis=1)
# testx=np.split(testdata,(1,),axis=1)
# print(testx)
# print(x)
# print(y)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.8)

# KFold
kf = KFold(n_splits=5, shuffle=True)

# print(kf.get_n_splits(x,y))

# for gamma in np.arange(0.1,10,0.1):

for train_index, test_index in kf.split(x):
    # print('train_index', train_index, 'test_index', test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(x_train,x_test)

    # clf svm
    # for c in np.arange(0.1,0.5,0.1):
    #     for gamma in np.arange(1,10,1):
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.0001, decision_function_shape='ovo')
    # clf=svm.LinearSVC(C=0.8, multi_class='ovr')
    clf.fit(x_train, y_train.ravel())
    # print(clf.score(x_train, y_train))
    y_that = clf.predict(x_train)
    tracc = sklearn.metrics.accuracy_score(y_that, y_train)
    print('Training Set Accuracy When gamma=', 'C=', 'is', tracc)
    # print(clf.score(x_test, y_test))
    y_testhat = clf.predict(x_test)
    teacc = sklearn.metrics.accuracy_score(y_testhat, y_test)
    print('Test Set Accuracy When gamma=', 'C=', 'is', teacc)

    # y_testdata = clf.predict(testdata)
    # print(y_testdata)
    #
    # decision_train = clf.decision_function(x_train)
    # predict_train = clf.predict(x_train)
    #
    # print('decision_function:\n', clf.decision_function(x_train))
    # print('\npredict:\n', clf.predict(x_train))
testdata = np.loadtxt('./dataset/test.txt', delimiter=':')
y_testdata = clf.predict(testdata)
print(y_testdata)
joblib.dump(clf, 'svm.m')

'''#clf
clf=svm.SVC(C=0.8,kernel='rbf',gamma=10,decision_function_shape='ovo')
clf.fit(x_train,y_train.ravel())
print(clf.score(x_train,y_train))
y_that=clf.predict(x_train)
tracc=sklearn.metrics.accuracy_score(y_that,y_train)
print('Training Set Accuracy',tracc)
print(clf.score(x_test,y_test))
y_testhat=clf.predict(x_test)
teacc=sklearn.metrics.accuracy_score(y_testhat,y_test)
print('Test Set Accuracy',teacc)'''