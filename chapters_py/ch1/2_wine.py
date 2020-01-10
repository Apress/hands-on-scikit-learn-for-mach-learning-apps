from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    br = '\n'
    data = load_wine()
    keys = data.keys()
    print (keys, br)
    X, y = data.data, data.target
    print ('features:', X.shape)
    print ('targets', y.shape, br)
    print (X[0], br)
    features = data.feature_names
    targets = data.target_names
    print ('feature set:')
    print (features, br)
    print ('targets:')
    print (targets, br)
    rnd_clf = RandomForestClassifier(random_state=0,
                                     n_estimators=100)
    rnd_clf.fit(X, y)
    rnd_name = rnd_clf.__class__.__name__
    feature_importances = rnd_clf.feature_importances_
    importance = sorted(zip(feature_importances, features),
                        reverse=True)
    n = 6
    print (n, 'most important features' + ' (' + rnd_name + '):')
    [print (row) for i, row in enumerate(importance) if i < n]