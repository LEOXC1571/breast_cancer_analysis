# breast_cancer_analysis

1. dim_reduce.py：数据初始化及模型调用
2. models.py：PCA， KPCA和LLE降维模型
3. plot.py：图
4. classify.py：运用SVM实现分类



## Workflow

本文的实验降维方法实现基于scikit-learn。

### 数据读取及预处理

- 原始数据为569条30维数据，分别代表30维特征。
- 30条特征分别为：mean radius，mean texture，mean perimeter，mean area，mean smoothness，mean compactness，mean concavity，mean concave points，mean symmetry，mean fractal dimension，radius error，texture error，perimeter error，area error，smoothness error，compactness error，concavity error，concave points error，symmetry error，fractal dimension error，worst radius，worst texture，worst perimeter，worst area，worst smoothness，worst compactness，worst concavity，worst concave points，worst symmetry，worst fractal dimension。
- 每条数据对应有一个0-1标签，0代表恶性，1代表良性。569条样本中有212个样本为恶性，357个为阳性。

### PCA降维

- 本文首先对30维的样本特征数据进行PCA降维，根据累计方差占比的折线图发现，主成分个数取2时对应因子累计方差占比的一个拐点，取6时对应另一个拐点，此时累计解释方差占比为88.76%。
- 本文给出了所有样本点位于解释方差占比前二及前三的空间内的分布图，并依据数据的标签予以上色。

### KPCA降维

- 接下来本文采取KPCA降维，考察其降维的效果。
- 核函数选择：由于Kernel-PCA方法主要是针对存在非线性关系的数据，故其核函数考虑线性核函数。
- 主成分个数：本文根据上文经验，考虑主成分个数3-10个。
- $\gamma$：该参数仅针对部分核函数的KPCA，在scikit-learn中sklearn.decompisition.KernelPCA中其默认取值为特征维数的倒数。
- 本文对上述三个参数进行超参数搜索，以获得最优的取值。
- 核函数的搜索范围为：[linear, rbf, poly, sigmoid, cosine]，主成分个数的搜索范围为[3，4，5，……，10]，$\gamma$的搜索范围为[0.003, 0.3]，进行10等分。
- 超参数结果及可视化见下文。

### LLE降维

- 本文采取LLE降维，考察其降维的效果。LLE假设具有非线性的样本点局部是线性的。
- 近邻个数：LLE降维对于近邻个数十分敏感，该参数设定需要非常小心。
- 主成分个数：为LLE降维后数据维度。
- 近邻算法：默认采用线性函数，本文认为者符合该数据集特点，同时其也有一些别的选择。
- 近邻个数搜索范围为：[3，5，7，10，20，30，50，100]，主成分个数搜索范围为：[2，3，4，……，10]，近邻算法考虑：[standard，hessian，modified，ltsa]

### 超参数实验

- 上述KPCA降维及LLE降维需要进行超参数实验。
- 为对比每组参数在breast cancer数据集上的表现优劣，本文运用SVM对样本分类进行训练及预测，根据其得分选取最优超参数。
- SVM其本身具备多个超参数，故在每个超参数实验中都同时对SVM的超参数进行搜索。
- SVM的超参数为C为正则化参数，用于防止过拟合，搜索范围为[0.1，0.2，……，0.9]。另一个超参数为SVM核函数的选择，搜索范围为[linear，rbf，poly，sigmoid]。
- 本文训练集与训练集划分比例为[0.8，0.2]。考虑本文数据集较小，故采取Kfold的方法，对数据集进行随机划分并训练5轮，以得到最优超参数。
- KPCA降维的最优参数为：主成分个数-9，KPCA核函数-linear，$\gamma$-0.003，正则化参数C-0.4，SVM核函数-linear。
- LLE降维的最优参数：主成分个数-7，近邻个数-3，LLE近邻算法-Standard（线性），正则化参数C-0.9，SVM核函数-linear。

### 可视化

- 本文给出KPCA和LLE各自在最优参数下在前二和前三主成分空间内的可视化情况。

### 分类模型

- 基于上文SVM的最优参数，本文对原始数据和经过三次降维后的数据进行分类模型的训练及预测。
- 原始数据：

Training Set Accuracy of linear kernel is: 0.967033, 0.971429, 0.969231, 0.967033, 0.960526
Test Set Accuracy of linear kernel is: 0.938596, 0.929825, 0.938596, 0.973684, 0.973451
The score of linear kernel is: 0.938596, 0.929825, 0.938596, 0.973684, 0.973451

- PCA降维：

Training Set Accuracy of linear kernel is:  0.9758241758241758, 0.9758241758241758, 0.9736263736263736, 0.9758241758241758, 0.9780701754385965
Test Set Accuracy of linear kernel is:  0.9736842105263158, 0.956140350877193, 0.9736842105263158, 0.956140350877193, 0.9646017699115044
The score of linear kernel is:  0.9736842105263158, 0.956140350877193, 0.9736842105263158, 0.956140350877193, 0.9646017699115044

- KPCA降维

Training Set Accuracy of linear kernel is:  0.9824175824175824, 0.9912087912087912, 0.9824175824175824, 0.9868131868131869, 0.9868421052631579
Test Set Accuracy of linear kernel is:  0.9912280701754386, 0.956140350877193, 0.9912280701754386, 0.9736842105263158, 0.9734513274336283
The score of linear kernel is:  0.9912280701754386, 0.956140350877193, 0.9912280701754386, 0.9736842105263158, 0.9734513274336283

- LLE降维

Training Set Accuracy of linear kernel is:  0.8285714285714286, 0.8571428571428571, 0.8263736263736263, 0.789010989010989, 0.8574561403508771
Test Set Accuracy of linear kernel is:  0.7982456140350878, 0.8333333333333334, 0.8596491228070176, 0.7368421052631579, 0.8938053097345132
The score of linear kernel is:  0.7982456140350878, 0.8333333333333334, 0.8596491228070176, 0.7368421052631579, 0.8938053097345132