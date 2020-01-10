import pandas as pd, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from random import randint

if __name__ == "__main__":
    br = '\n'
    tips = pd.read_csv('data/tips.csv')
    data = tips.drop(['tip'], axis=1)
    target = tips['tip']
    v = ['sex', 'smoker', 'day', 'time']
    ls = data[v].to_dict(orient='records')
    vector = DictVectorizer(sparse=False, dtype=int)
    d = vector.fit_transform(ls)
    print ('one hot encoding:')
    print (d[0:3], br)
    print ('encoding order:')
    encode_order = vector.get_feature_names()
    print (encode_order, br)
    data = data.drop(['sex', 'smoker', 'day', 'time'], axis=1)
    X = data.values
    print ('feature shape after removing categorical columns:')
    print (X.shape, br)
    Xls, dls = X.tolist(), d.tolist()
    X = [np.array(row + dls[i]) for i, row in enumerate(Xls)]
    X = np.array(X)
    y = target.values
    print ('feature shape after adding encoded data back:')
    print (X.shape, br)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    model = LinearRegression(fit_intercept=True)
    model_name = model.__class__.__name__
    print ('<<' + model_name  +  '>>', br)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print (rmse, '(rmse)', br)
    print ('predict 1st test set element (actual/prediction):')
    print (y_test[0], y_pred[0], br)
    rints = [randint(0, y.shape[0]-1) for row in range(3)]
    print ('random integers:', rints, br)
    p = [X[rints[0]], X[rints[1]], X[rints[2]]]
    y_p = model.predict(p)
    y_p = list(np.around(y_p, 2))
    print (y_p, '(predicted)')
    print ([y[rints[0]], y[rints[1]], y[rints[2]]], '(actual)')