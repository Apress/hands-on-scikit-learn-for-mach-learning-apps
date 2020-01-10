# install humanfriendly if necessary

import numpy as np, humanfriendly as hf, random
import time
from sklearn.model_selection import train_test_split,\
     RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def get_scores(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    name = model.__class__.__name__
    return (name, train, test)

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

def prep_data(data, target):
    d = [data[i] for i, _ in enumerate(data)]
    t = [target[i] for i, _ in enumerate(target)]
    return list(zip(d, t))

def create_sample(d, n, replace='yes'):
    if replace == 'yes': s = random.sample(d, n)
    else: s = [random.choice(d)
               for i, _ in enumerate(d) if i < n]
    Xs = [row[0] for i, row in enumerate(s)]
    ys = [row[1] for i, row in enumerate(s)]
    return np.array(Xs), np.array(ys)

def see_time(note):
    end = time.perf_counter()
    elapsed = end - start
    print (note,
           hf.format_timespan(elapsed, detailed=True))

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_bank.npy')
    # need to add allow_pickle=True parameter
    y = np.load('data/y_bank.npy', allow_pickle=True)
    sample_size = 4000
    data = prep_data(X, y)
    Xs, ys = create_sample(data, sample_size, replace='no')
    X_train, X_test, y_train, y_test = train_test_split\
                                       (Xs, ys, random_state=0)
    knn = KNeighborsClassifier()
    print (knn, br)
    knn.fit(X_train, y_train)
    knn_scores = get_scores(knn, X_train, y_train,
                            X_test, y_test)
    print (knn_scores[0] + ' (train, test):')
    print (knn_scores[1], knn_scores[2], br)
    param_grid = {'n_neighbors': np.arange(1, 31, 2),
                  'metric': ['euclidean']}
    start = time.perf_counter()
    rand = RandomizedSearchCV(knn, param_grid, cv=3, n_jobs = -1,
                              random_state=0, verbose=2)
    rand.fit(X, y)
    see_time('RandomizedSearchCV total tuning time:')
    bp = rand.best_params_
    print (bp, br)
    file = 'data/bp_bank'
    np.save(file, bp)
    knn_tuned = KNeighborsClassifier(**bp).fit(X_train, y_train)
    knn_scores = get_scores(knn_tuned, X_train, y_train,
                            X_test, y_test)
    print (knn_scores[0] + ' (train, test):')
    print (knn_scores[1], knn_scores[2], br)
    print ('cross-validation score:')
    knn = KNeighborsClassifier()
    scores = get_cross(knn, Xs, ys)
    print (np.mean(scores))
