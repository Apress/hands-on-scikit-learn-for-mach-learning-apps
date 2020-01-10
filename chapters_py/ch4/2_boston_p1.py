import numpy as np, pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    br = '\n'
    boston = load_boston()
    X = boston.data
    y = boston.target
    print ('feature shape', X.shape)
    print ('target shape', y.shape, br)
    keys = boston.keys()
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=100)
    rfr.fit(X, y)
    features = boston.feature_names
    feature_importances = rfr.feature_importances_
    importance = sorted(zip(feature_importances, features),
                        reverse=True)
    [print (row) for row in importance]
    print ()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=100)
    rfr.fit(X_train, y_train)
    rfr_name = rfr.__class__.__name__
    y_pred = rfr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print (rfr_name + ' (rmse):', rmse, br)
    cols = list(features) + ['target']
    data = pd.DataFrame(data=np.c_[X, y], columns=cols)
    print ('boston dataset sample:')
    print (data[['RM', 'LSTAT', 'DIS', 'CRIM', 'NOX', 'PTRATIO',
                 'target']].head(3), br)
    print ('data set before removing noise:', data.shape)
    noise = data.loc[data['target'] >= 50]
    data = data.drop(noise.index)
    print ('data set without noise:', data.shape, br)
    X = data.loc[:, data.columns != 'target'].values
    y = data['target'].values
    print ('cleansed feature shape:', X.shape)
    print ('cleansed target shape:', y.shape, br)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=100)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print (rfr_name + ' (rmse):', rmse)
    X_file = 'data/X_boston'
    y_file = 'data/y_boston'
    np.save(X_file, X)
    np.save(y_file, y)