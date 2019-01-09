import pandas as pd
import numpy as np
from sklearn import preprocessing
from regBoost import RegBoost


def read_data(filename):
    data = pd.read_csv(filename).sample(frac=1)
    # 打乱数据
    Xs = data[["F1","F2","F3","F4","F5","F6", "F7", "F8", "F9"]].values.astype(float)
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
    ypreds = []
    for i in range(10):
        model = RegBoost(trainX, trainY)
        model.fit()
        ypreds.append(model.predict(testX, testY))
    ypreds = np.array(ypreds)
    instances = [1,12,35,90,130]
    ypred = np.mean(ypreds, axis=0) # 按列求平均

    print('##-----------------------------------------')
    print(ypreds[:,instances])
    print('@@-----------------------------------------')
    for i in instances:
        print(ypred[i], testY[i])
    print('!!-----------------------------------------')

    #3 模型评价
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print(mse, mse**0.5, mae)
