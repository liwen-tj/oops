import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing


def read_data(filename):
    # 打乱数据
    # data = pd.read_csv(filename).sample(frac=1)
    data = pd.read_csv(filename)
    
    # X_scaled（二维数组）, Ys（一维数组）都是ndarray
    Ys = data.critical_temp.values
    Xs = data.drop(columns=['critical_temp']).values
    B = 20000 # Train Num
    T = 21263
    (trainX, testX) = (Xs[:B], Xs[B:T])
    (trainY, testY) = (Ys[:B], Ys[B:T])
    return (trainX, testX, trainY, testY)

def train(trainX, trainY):
	# 模型
    params = {
        'learning_rate': 0.1,
        'objective': 'regression_l2',
        'max_depth': 4,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'metric': 'mse',
        'seed': 2019
    }
    train_data = lgb.Dataset(trainX, label=trainY)
    bst = lgb.train(params, train_data, num_boost_round=200)
    return bst


if __name__ == '__main__':
    (trainX, testX, trainY, testY) = read_data("train.csv")
    bst = train(trainX, trainY)
    ypred = bst.predict(testX)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(testY, ypred)
    mae = mean_absolute_error(testY, ypred)
    print('----------------------')
    print(mse, mse**0.5, mae)
    
