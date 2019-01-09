import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing


def read_data(filename):
    data = pd.read_csv(filename).sample(frac=1)
    # 打乱数据
    Xs = data[["F1","F2","F3","F4","F5","F6", "F7", "F8", "F9"]].values.astype(float)
    X_scaled = preprocessing.MinMaxScaler().fit_transform(Xs)
    Ys = data.RMSD.values
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    
    B = 40000 # Train Num
    T = 45730
    (trainX, testX) = (X_scaled[:B], X_scaled[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)


def train(trainX, trainY):
	# 模型
    params = {
        'learning_rate': 0.1,
        'objective': 'regression_l1',
        'max_depth': 3,
        'bagging_fraction': 0.5,
        'min_data_in_leaf': 50,
        'metric': 'mse',
        'seed': 42
    }
    train_data = lgb.Dataset(trainX, label=trainY)
    bst = lgb.train(params, train_data, num_boost_round=100)
    return bst


if __name__ == '__main__':
    # instances = 45730
    (trainX, testX, trainY, testY) = read_data("CASP.csv")
    bst = train(trainX, trainY)
    ypred = bst.predict(testX)
    for i in range(100):
        print(ypred[i], testY[i])
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print(mse, mae)
    
