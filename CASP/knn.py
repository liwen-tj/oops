import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename).sample(frac=1)
    # 打乱数据
    Xs = data[["F1","F2","F3","F4","F5","F6","F7","F8","F9"]].values.astype(float)
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    Ys = data.RMSD.values
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    
    B = 40000 # Train Num
    T = 45730
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)


if __name__ == '__main__':
    (trainX, testX, trainY, testY) = read_data("CASP.csv")
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(trainX)
    (dist, ind) = neigh.kneighbors(testX)

    ypred = np.array([(trainY[i[0]] + trainY[i[1]] + trainY[i[2]]) / 3.0 for i in ind])

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print(mse, mse**0.5, mae)
