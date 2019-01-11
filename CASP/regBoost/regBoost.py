import random
import heapq
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import f_regression


class RegBoost:
    feature_num = 9     # 每个线性回归的特征数目
    learning_rate = 0.1 # 学习速率
    max_layer = 15      # 最大递归次数（最多经过的学习器个数）
    min_sample = 20     # 数目少于min_sample个时则不再进行线性回归
    knn = 3             # K近邻中的参数K
    bagging_fraction = 1 # 对样本进行采样
    feature_fraction = 1 # 对特征进行采样
    features = 9          # 总特征数目


    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY


    def select_features(self, X):
        '''X是样本索引列表'''
        gains, _ = f_regression(self.trainX[X,:], np.array([self.trainY[i] for i in X]))
        # 特征采样
        for i in range(self.features):
            if random.random() > self.feature_fraction:
                gains[i] = -1
        
        # 最大的self.feature_num个数的索引
        # max_gains_id = map(gains.index, heapq.nlargest(self.feature_num, gains))
        max_gains_id = heapq.nlargest(self.feature_num, range(len(gains)), gains.take)
        sid = list(max_gains_id)
        # print('selected feature id =', sid)
        return sid


    def get_model(self, Xs, layer):
        '''Xs: 样本在训练集中的索引'''
        print('layer =', layer)
        #0 根据bagging_fraction对样本采样
        X = []
        for x in Xs:
            if random.random() < self.bagging_fraction:
                X.append(x)
        #1 选择信息增益最大的feature_num个特征
        best_features = self.select_features(X)
        dataX = self.trainX[X,:][:,best_features]
        dataY = np.array([self.trainY[i] for i in X])
        #2 执行回归操作
        regr = linear_model.LinearRegression()
        regr.fit(dataX, dataY)
        ypreds = regr.predict(self.trainX[Xs,:][:,best_features]) # 对所有数据进行预测，不仅仅是对样本进行采样后的数据

        #3 分出正负组，并修改Y值
        Xp, Xn = [], []
        i = 0
        for xs in Xs:
            if self.trainY[xs] >= ypreds[i]:
                Xp.append(xs)
            else:
                Xn.append(xs)
            self.trainY[xs] = self.trainY[xs] - self.learning_rate * ypreds[i] # 修正Y值       
            i += 1
        
        #4 kd tree
        neigh = NearestNeighbors(n_neighbors=self.knn)
        data = self.trainX[Xp+Xn,:][:,best_features]
        neigh.fit(data)
        
        #5 正负类模型
        Xp, Xn = np.array(Xp), np.array(Xn)
        model_p = self.get_model(Xp, layer-1) if layer > 0 and len(Xp) > self.min_sample else None
        model_n = self.get_model(Xn, layer-1) if layer > 0 and len(Xn) > self.min_sample else None

        model = (regr, best_features, neigh, len(Xp), model_p, model_n)
        return model


    def train(self):
        self.model = self.get_model(range(len(self.trainX)), self.max_layer)
        print("...train done...")
    

    def test(self, testX):
        results = []
        for tx in testX:
            preds = []
            model = self.model
            while model is not None:
                txx = np.array([[tx[f] for f in model[1]]])
                # model[0]:regr    model[1]:best_features
                preds.append(model[0].predict(txx)[0])
                # model[2]:neigh
                (dist, ind) = model[2].kneighbors(txx)
                (dist, ind) = (dist[0], ind[0])
                # model[3]:len(Xp)
                kp, kn = 0, 0
                for i in ind:
                    if i < model[3]:
                        kp += 1
                    else:
                        kn += 1
                model = model[4] if kp > kn else model[5]
            results.append(sum(preds[:-1])*self.learning_rate + preds[-1])
        return results
