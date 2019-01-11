import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import numpy as np


def read_data(filename):
    # 打乱数据
    data = pd.read_csv(filename).sample(frac=1)
    
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    Ys = data.critical_temp.values
    Xs = data.drop(columns=['critical_temp']).values
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    B = 20000 # Train Num
    T = 21263
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)


if __name__ == '__main__':
    (trainX, testX, trainY, testY) = read_data("train.csv")
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(trainX)
    (dist, ind) = neigh.kneighbors(testX)

    # ypred = np.array([(trainY[i[0]] + trainY[i[1]] + trainY[i[2]]) / 3.0 for i in ind])
    ypred = np.array([trainY[i[0]] for i in ind])

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print('mse =', mse)
    print('rmse =', mse**0.5)
    print('mae =', mae)
