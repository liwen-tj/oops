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
    data = pd.read_csv(filename)#.sample(frac=1)
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    Ys = data.critical_temp.values
    Xs = data.drop(columns=['critical_temp']).values
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    B = 19000 # Train Num
    T = 21263
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)


if __name__ == '__main__':
    (trainX, testX, trainY, testY) = read_data("../train.csv")
    # train a model
    models = []
    for r in range(50):
        print('r =', r+1)
        model = RegBoost(trainX, trainY)
        model.train()
        models.append(model)
        (trainX, testX, trainY, testY) = read_data("../train.csv")
        ys = []
        for m in models:
            ys.append(m.test(trainX))
        ys = np.array(ys)
        ys = 0.1 * np.sum(ys, axis=0)
        trainY = np.array([trainY[i] - ys[i] for i in range(19000)])
        print(trainY[:10])
        print('\n')
        
    # test
    ypreds = []
    for m in models:
        ypreds.append(m.test(testX))
    ypreds = np.array(ypreds)
    # ypred = np.sum(ypreds, axis=0) # 按列求平均
    ypred = 0.1 * np.sum(ypreds[range(49),:], axis=0) + ypreds[-1]

    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print('mse =', mse)
    print('rmse =', mse**0.5)
    print('mae =', mae)
