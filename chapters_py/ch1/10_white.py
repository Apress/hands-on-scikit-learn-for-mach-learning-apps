import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    br = '\n'
    f = 'data/whitewine.csv'
    white_wine = pd.read_csv(f)
    X = white_wine.drop(['quality'], axis=1)
    y = white_wine['quality']
    print (X.shape)
    print (y.shape, br)
    features = list(X)
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=100)
    rfr_name = rfr.__class__.__name__
    rfr.fit(X, y)
    feature_importances = rfr.feature_importances_
    importance = sorted(zip(feature_importances, features),
                        reverse=True)
    n = 3
    print (n, 'most important features' + ' (' + rfr_name + '):')
    [print (row) for i, row in enumerate(importance) if i < n]
    print ()
    print (white_wine[['alcohol', 'sulphates',
                       'volatile acidity',
                       'total sulfur dioxide',
                       'quality']]. head())