import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#定义fisher分类器类
class FisherClassifier:
    #初始化参数
    def __init__(self):
        self.w = None #投影方向
        self.threshold = None #类别临界值
    #训练模型
    def fit(self,X,y):
        # x:特征矩阵，每一行是一个样本，每一列是一个特征
        # y:标签向量，每个元素是一个样本的类别

        # 获取样本数和特征数
        n_samples,n_features = X.shape

        #获取类别数和类别列表
        n_classes = len(np.unique(y))
        classes = np.unique(y)

        #计算每个类别的样本均值项链和总体均值向量
        mean_vectors = []
        mean_all = np.mean(X,axis=0)
        for c in classes:
            mean_vectors.append(np.mean(X[y==c],axis=0))

        #计算每个类别的样本协方差矩阵和总体协方差矩阵
        cov_matrices = []
        cov_all = np.cov(X.T)
        for c in classes:
            cov_matrices.append(np.cov(X[y==c].T))

        #计算类间散度矩阵和类内散度矩阵
        S_B = np.zeros((n_features,n_features))
        S_W = np.zeros((n_features,n_features))
        for i,mean_vec in enumerate(mean_vectors):
            n_i = X[y==classes[i]].shape[0]
            mean_vec = mean_vec.reshape(n_features,1)
            mean_all = mean_all.reshape((n_features,1))
            S_B += n_i*(mean_vec-mean_all).dot((mean_vec-mean_all).T)
            S_W += cov_matrices[i]

        #求解类间散度矩阵和类内散度矩阵的广义特征值问题，得到最优的投影方向
        eig_vals,eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        self.w = eig_vecs[:np.argmax(eig_vals)]

        #将原始数据投影到最优方向上，得到新的特征空间
        X_new = X.dot(self.w)

        #在新的特征空间中，建立判别函数，根据样本在最优方向上的投影值，计算类别临界值
        #假设两个类别的投影值服从正态分布，且方差相等，则类别临界值为两个均值的中点
        mean_new = []
        for c in classes:
            mean_new.append(np.mean(X_new[y==c]))
            self.threshold = np.mean(mean_new)

        #对已知类别的样本进行判别归类，评估分类准确率
        y_pred = np.where(X_new > self.threshold,classes[1],classes[0])
        accuracy = np.mean(y_pred ==y)
        print("Traning accuravy:",accuracy)

        #预测新样本的类别
        def predict(self,X):
            # X:特征矩阵，每一行是一个样本，每一列是一个特征

            # 将新样本投影到最优方向上
            X_new = X.dot(self.w)

            #根据类别临界值进行判别
            y_pred = np.where(X_new > self.threshold,classes[1],classes[0])

            #返回预测结果
            return y_pred

if __name__ =="__main__":
    df = pd.read_excel("datasets.xlsx")
    X = df.iloc[:,1:4].values  #特征矩阵
    y = df.iloc[:,4].values    #标签向量

    #划分训练集和测试集
    X_train = X[:10]           #前10个样本作为训练集
    y_train = y[:10]
    X_test = X[10:]            #后4个样本作为测试集
    y_test = y[10:]

    #创建并训练fisher分类器
    clf = FisherClassifier()
    clf.fit(X_train,y_train)

    #预测测试集的类别
    y_pred = clf.predict(X_test)
    print(y_pred)