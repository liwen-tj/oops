import random
import heapq
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors

class RegBoost:
    feature_num = 2     # 每个线性回归的特征数目
    learning_rate = 0.1 # 学习速率
    max_layer = 10      # 最大递归次数（最多经过的学习器个数）
    min_sample = 30     # 数目少于min_sample个时则不再进行线性回归
    knn = 9             # K近邻中的参数K
    bagging_fraction = 0.8 # 对样本进行采样
    feature_fraction = 0.6 # 对特征进行采样
    features = 81          # 总特征数目


    def select_features(self, samples):
        gains = []
        for i in range(self.features):
            if random.random() > self.feature_fraction: # 特征采样 
                gains.append(-1)
            else: # 被抽中，正常计算其gain
                gains.append(random.random()+i) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO DO
        # 最大的self.feature_num个数的索引
        max_gains_id = map(gains.index, heapq.nlargest(self.feature_num, gains))
        return list(max_gains_id)


    def get_model(self, Xs, layer):
        '''Xs: 样本在训练集中的索引'''
        #0 根据bagging_fraction对样本采样
        X = []
        for x in Xs:
            if random.random() < self.bagging_fraction:
                X.append(x)
        samples = self.trainX[X,:]
        dataY = np.array([self.trainY[i] for i in X])

        #1 选择信息增益最大的feature_num个特征
        best_features = self.select_features(samples)
        dataX = samples[:,best_features]

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


    def train(self, trainX, trainY):
        (self.trainX, self.trainY) = (trainX, trainY)
        self.model = self.get_model(range(len(trainX)), self.max_layer)
        print("...train done...")
    

    def test(self, testX):
        model = self.model
        results = []
        for tx in testX:
            preds = []
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
            results.append(sum(preds[:-1] + preds[-1]))
        return results
        