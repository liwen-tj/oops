import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from regBoost import RegBoost
from sklearn import preprocessing
import numpy as np
import imp


def read_data(filename):
    # 打乱数据
    #data = pd.read_csv(filename).sample(frac=1)
    data = pd.read_csv(filename)
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
    ypreds = []
    for _ in range(20):
        (trainX, testX, trainY, testY) = read_data("../train.csv")
        print(trainY[100:105])
        model = RegBoost(trainX, trainY)
        model.train()
        y = model.test(testX)
        ypreds.append(y)
        print(trainY[100:105])
        print('\n\n\n')
        

    ypreds = np.array(ypreds)
    instances = [1,12,35,90,130, 210,220,245,890,1000]
    print(ypreds[:,instances])
    print("====================")
    print([testY[i] for i in instances])
    print('------------------------------------------------')

    ypred = np.mean(ypreds, axis=0) # 按列求平均

    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print('mse =', mse)
    print('rmse =', mse**0.5)
    print('mae =', mae)
