from random import sample
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import numpy as np


class RegBst2:
    feature_num = 2 # 每个线性回归的特征数目
    learner_num = 10 # 有多少个学习器可供选择
    max_learner = 2 # 至多选择max_learner个学习器
    learning_rate = 0.2 # 学习速率
    max_layer = 10 # 最大递归次数（最多经过的学习器个数）
    min_sample = 200 # 数目少于min_sample个时则不再进行线性回归
    Knn = 11
    times_temp = 0
    def __init__(self, train_data, labels):
        self.train_data = train_data # shape = (instances, attributes)
        self.labels = labels
        self.attribute_num = len(train_data[0])


    def get_feature(self):
        '''选择特征组，shape=(learner_num, feature_num)'''
        features = set() # 集合类型数据
        while features.__len__() != self.learner_num:
            c = tuple(sorted(sample(range(self.attribute_num), self.feature_num)))
            features.add(c) # 集合元素不可以add列表类型的数据，所以上一步转为tuple
        return list(features) # 集合类型不可以索引，还要再转成列表类型


    def get_learner(self, fea, data, Y):
        '''根据选定的特征，还有数据，得到基学习器'''
        """!!! 注意修改Y值 !!! fea, data, Y都是list数据"""
        #1 取出数据
        dataX = self.train_data[:,fea][data,:]
        #2 基学习器
        regr = linear_model.LinearRegression()
        regr.fit(dataX, Y)
        pred = regr.predict(dataX)
        #3 分出正负组，并修改Y值
        n = data.__len__()
        data_p, data_n, Yp, Yn = [], [], [], []
        for i in range(n):
            if(pred[i] > Y[i]):
                data_p.append(data[i])
                Yp.append(Y[i] - self.learning_rate*pred[i])
            else:
                data_n.append(data[i])
                Yn.append(Y[i] - self.learning_rate*pred[i])
        return (regr, data_p, data_n, Yp, Yn)

    # TO DO
    def select_learners(self, learners):
        '''
        输入： candidate learners
        输出： 选定的学习器
        '''
        
        return []


    def get_model(self, data, Y, layer):
        print('get_model layer =', layer, 'times =', self.times_temp)
        self.times_temp += 1
        # 选择特征
        features = self.get_feature()
        # 可供选择的学习器
        cand_learners = []
        for i in range(self.learner_num):
            fea = features[i]
            (learner, data_p, data_n, Yp, Yn) = self.get_learner(fea, data, Y)
            # data_p, data_n, Yp, Yn都是list数据类型
            cand_learners.append((learner, data_p, data_n, Yp, Yn, fea)) ### 后面的lr
        # 选定学习器，继续向下扩展
        learners = self.select_learners(cand_learners) # 一定不为空
        model = []
        for lr in learners:
            if(layer <= 1 or lr[1].__len__() < self.min_sample): # 已经达到最大层数
                print('终止 layer =', layer, '  samples =', lr[1].__len__())
                model_p = None
            else:
                model_p = self.get_model(lr[1], lr[3], layer-1)
            
            if(layer <= 1 or lr[2].__len__() < self.min_sample): # 已经达到最大层数
                print('终止 layer =', layer, '  samples =', lr[2].__len__())
                model_n = None
            else:
                model_n = self.get_model(lr[2], lr[4], layer-1)
            
            # lr[0], lr[5], lr[1], lr[2]分别是指：learner, feature, data_p, data_n
            model.append((lr[0], lr[5], lr[1], lr[2], model_p, model_n))
        return model


    def fit(self):
        data_id = range(len(self.train_data))
        # 包含了所有的基学习器，还有每个基学习器分出的数据索引
        self.model = self.get_model(data_id, self.labels, self.max_layer)
        print('fit done...')
    

    def pn_samples_dis(self, td, fea, datap, datan):
        '''计算datap和datan所有数据，最近的K个在正负数据类中的分布'''
        (kp, kn) = (0, 0)
        #1 联合所有数据，计算最近的K个
        data = datap + datan
        samples = self.train_data[:,fea][data,:]
        neigh = NearestNeighbors(n_neighbors=self.Knn)
        neigh.fit(samples)
        (dist, ind) = neigh.kneighbors([[td[f] for f in fea]])
        (dist, ind) = (dist[0], ind[0])
        #2 确定这K个最近点的分布如何
        for i in ind:
            if i < datap.__len__():
                kp += 1
            else:
                kn += 1
        #3 Done
        #print("正负值：", kp, kn)
        return kp - kn


    def select_best_learner(self, model, td):
        '''计算该层模型中最合适的基学习器'''
        (idx, pos, maxd) = (0, 0, 0)
        for i in range(self.learner_num):
            # 计算正负差
            d = self.pn_samples_dis(td, model[i][1], model[i][2], model[i][3])
            if abs(d) > maxd:
                (idx, pos, maxd) = (i, 0 if d > 0 else 1, abs(d))
        #print('max distance =', maxd)
        return (idx, pos)

    
    def predict(self, test_data, testY):
        '''test_data(shape=(instances, attributes))是ndarray，返回值也应该是ndarray'''
        results = []
        y = 0
        for td in test_data:
            pred = []
            model_curre = self.model
            while True:
                #1 选定基学习器 返回的是学习器的idx(list元素的index)和下一个学习器的正(0)负(1)
                (idx, pos) = self.select_best_learner(model_curre, td)
                #2 计算当前基学习器的预测值
                regr = model_curre[idx][0]
                feas = model_curre[idx][1]
                y_t = regr.predict(np.array([[td[i] for i in feas]]))[0]
                #3 判断下一层是否为空，空则结束
                model_curre = model_curre[idx][4+pos] # pos正(0)负(1)
                if model_curre is not None:
                    pred.append(y_t * self.learning_rate)
                else:
                    pred.append(y_t)
                    break
            print(sum(pred), '\t', testY[y])
            results.append(sum(pred))
            y += 1
    
        return results
    
