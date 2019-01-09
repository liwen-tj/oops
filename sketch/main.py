import pandas as pd
import numpy as np
from sklearn import preprocessing
from regBoost import RegBoost


def read_data(filename):
    data = pd.read_csv(filename).sample(frac=1)
    # 打乱数据
    Xs = data[["F1","F2","F3","F4","F5","F6"]].values.astype(float)
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    Ys = data.RMSD.values
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    B = 45000 # Train Num
    T = 45300
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)


if __name__ == '__main__':
    # instances = 45730
    (trainX, testX, trainY, testY) = read_data("CASP.csv")
    #1 模型搭建
    model = RegBoost(trainX, trainY)
    model.fit()
    #2 模型预测
    ypred = model.predict(testX, testY)
    #3 模型评价
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print(mse, mse**0.5, mae)
