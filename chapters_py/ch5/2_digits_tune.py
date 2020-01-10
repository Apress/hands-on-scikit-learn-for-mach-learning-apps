# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,\
     cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    print (knn, br)
    train, test, name = get_scores(knn, X_train, y_train,
                                   X_test, y_test)
    knn_name, acc1, acc2 = name, train, test
    print (str(knn_name) + ':')
    print ('train:', np.round(acc1, 2),
           'test:', np.round(acc2, 2), br)
    param_grid = {'n_neighbors': np.arange(1, 31, 2),
                  'metric': ['euclidean', 'cityblock']}
    start = time.perf_counter()
    grid = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
    see_time('GridSearchCV total tuning time:')
    best_params = grid.best_params_
    print (best_params, br)
    knn_tuned = KNeighborsClassifier(**best_params)
    knn_tuned.fit(X_train, y_train)
    train, test, name = get_scores(knn_tuned, X_train, y_train,
                                   X_test, y_test)
    knn_name, acc1, acc2 = name, train, test
    print (knn_name + ' (tuned):')
    print ('train:', np.round(acc1, 2),
           'test:', np.round(acc2, 2), br)
    lr = LogisticRegression(random_state=0, max_iter=4000,
                            multi_class='auto', solver='lbfgs')
    print (lr, br)
    lr.fit(X_train, y_train)
    train, test, name = get_scores(lr, X_train, y_train,
                                   X_test, y_test)
    lr_name, acc1, acc2 = name, train, test
    print (lr_name + ':')
    print ('train:', np.round(acc1, 2),
           'test:', np.round(acc2, 2), br)
    param_grid = {'penalty': ['l2'],
                  'solver': ['newton-cg', 'lbfgs', 'sag'],
                  'max_iter': [4000], 'multi_class': ['auto'],
                  'C': [0.001, 0.01, 0.1]}
    start = time.perf_counter()
    grid = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
    see_time('GridSearchCV total tuning time:')
    bp = grid.best_params_
    print (bp)
    lr_tuned = LogisticRegression(**bp, random_state=0)
    lr_tuned.fit(X_train, y_train)
    train, test, name = get_scores(lr_tuned, X_train, y_train,
                                   X_test, y_test)
    lr_name, acc1, acc2 = name, train, test
    print (lr_name + ' (tuned):')
    print ('train:', np.round(acc1, 2),
           'test:', np.round(acc2, 2), br)
    print ('cross-validation score knn:')
    knn = KNeighborsClassifier()
    scores = get_cross(knn, X, y)
    print (np.mean(scores))