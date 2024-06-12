import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data
data = pd.read_csv("iris.data.csv", header=0)

# data.sample()
data["species"] = data["species"].map({"virginica": 0, "setosa": 1, "versicolor": 2})
data.duplicated().any()
# 删除重复参数
data.drop_duplicates(inplace=True)
len(data)
print(data["species"].value_counts())

class KNN:
    def __init__(self, k):
        """初始化方法"""
        self.k = k

    def fit(self, X, y):
        """ 训练方法
            Parameters
            ----------
            X：类数组类型，形状为【样本数量， 特征数量】
                待训练样本特征（属性）

            y：类数组类型，形状为：【样本数量】
            每隔样本的目标值（标签）
        """
        # 将X转换成ndarray数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """根据参数传递的样本，对样本数据进行预测

        Parameters
        --------
        X：类数组类型，形状为【样本数量， 特征数量】
                待训练样本特征（属性）
        returns
        --------
        result:数组类型
                预测的结果。
        """
        X = np.asarray(X)
        result = []
        # 对ndarray数组进行遍历，每次取数组中的一行
        for x in X:
            # 对于测试集中的每一个样本， 依次与训练金进行距离计算
            dis = np.sqrt(np.sum((x - self.X) ** 2), axis=1)
            index = dis.argsort()
            # 进行截断 只取前K个元素的指引
            index = index[:self.k]
            self.y[index]
            # 返回数组中没哥哥元素出现的次数 元素必须是非负的整数
            count = np.bincount(self.y[index])
            # 返回ndarray数组中，值最大的对应的索引
            # 最大元素就是出现最多的元素
            result.append(count.argmax())
        return np.array(result)





