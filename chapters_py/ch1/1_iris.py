from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    br = '\n'
    iris = datasets.load_iris()
    keys = iris.keys()
    print (keys, br)
    X = iris.data
    y = iris.target
    print ('features shape:', X.shape)
    print ('target shape:', y.shape, br)
    features = iris.feature_names
    targets = iris.target_names
    print ('feature set:')
    print (features, br)
    print ('targets:')
    print (targets, br)
    print (iris.DESCR[525:900], br)
    rnd_clf = RandomForestClassifier(random_state=0,
                                     n_estimators=100)
    rnd_clf.fit(X, y)
    rnd_name = rnd_clf.__class__.__name__
    feature_importances = rnd_clf.feature_importances_
    importance = sorted(zip(feature_importances, features),
                        reverse=True)
    print ('most important features' + ' (' + rnd_name + '):')
    [print (row) for i, row in enumerate(importance)]