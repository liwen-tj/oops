import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from regBoost import RegBoost
from sklearn import preprocessing
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename)#.sample(frac=1)
    # 打乱数据
    Ys = data.RMSD.values
    Xs = data.drop(columns=['RMSD']).values
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    B = 40000 # Train Num
    T = 45730
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)



if __name__ == '__main__':
    ypreds = []
    for _ in range(1):
        (trainX, testX, trainY, testY) = read_data("../CASP.csv")
        model = RegBoost(trainX, trainY)
        model.train()
        y = model.test(testX)
        ypreds.append(y)
        

    ypreds = np.array(ypreds)
    ypred = np.mean(ypreds, axis=0) # 按列求平均

    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print('mse =', mse)
    print('rmse =', mse**0.5)
    print('mae =', mae)
