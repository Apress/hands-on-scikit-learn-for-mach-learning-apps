from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    br = '\n'
    boston = load_boston()
    print (boston.keys(), br)
    print (boston.feature_names, br)
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
    [print(row) for i, row in enumerate(importance) if i < 3]