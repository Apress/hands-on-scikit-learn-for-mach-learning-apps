# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def get_scores(model, Xtrain, ytrain, Xtest, ytest):
    y_pred = model.predict(Xtrain)
    train = accuracy_score(ytrain, y_pred)
    y_pred = model.predict(Xtest)
    test = accuracy_score(ytest, y_pred)
    return train, test, model.__class__.__name__

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

def see_time(note):
    end = time.perf_counter()
    elapsed = end - start
    print (note,
           hf.format_timespan(elapsed, detailed=True))

if __name__ == "__main__":
    br = '\n'
    iris = load_iris()
    X = iris.data
    y = iris.target
    targets = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    knn = KNeighborsClassifier()
    print (knn, br)
    distances = [1, 2, 3, 4, 5]
    k_range = list(range(1, 31))
    leaf = [10]
    param_grid = dict(n_neighbors=k_range, p=distances,
                      leaf_size=leaf)
    start = time.perf_counter()
    grid = GridSearchCV(knn, param_grid, cv=10,
                        scoring='accuracy')
    grid.fit(X, y)
    see_time('GridSearchCV total tuning time:')
    bp = grid.best_params_
    print ()
    print ('best parameters:')
    print (bp, br)
    knn_best = KNeighborsClassifier(**bp).fit(X_train, y_train)
    train, test, name = get_scores(knn_best, X_train, y_train,
                                   X_test, y_test)
    print (name, 'train/test scores (GridSearchCV):')
    print (train, test, br)
    scores = get_cross(knn, X, y)
    print ('cross-val scores:')
    print (scores, br)
    print ('avg cross-val score:', np.mean(scores), br)
    d = grid.cv_results_
    print ('mean grid score:', np.mean(d['mean_test_score']), br)
    vector = [[3, 5, 4, 2]]
    vectors = [[2, 5, 3, 5], [1, 4, 2, 1]]
    y_pred = knn_best.predict(vector)
    print (targets[y_pred])
    y_preds = knn_best.predict(vectors)
    print (targets[y_preds], br)
    start = time.perf_counter()
    rand = RandomizedSearchCV(knn, param_grid, cv=10,
                              random_state=0,
                              scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    see_time('RandomizedSearchCV total tuning time:')
    bp = rand.best_params_
    print()
    print ('best parameters:')
    print (bp, br)
    knn_best = KNeighborsClassifier(**bp).fit(X_train, y_train)
    train, test, name = get_scores(knn_best, X_train, y_train,
                                   X_test, y_test)
    print (name, 'train/test scores (RandomizedSearchCV):')
    print (train, test)