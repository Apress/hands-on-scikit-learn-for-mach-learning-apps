# install humanfriendly if necessary

import numpy as np, humanfriendly as hf, random
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,\
     cross_val_score
from sklearn.ensemble import RandomForestClassifier,\
     ExtraTreesClassifier

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
    else: s = [random.choice(d) for i, _ in enumerate(d)
               if i < n]
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
    X_file = 'data/X_mnist'
    y_file = 'data/y_mnist'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    X = X.astype(np.float32)
    data = prep_data(X, y)
    sample_size = 7000
    Xs, ys = create_sample(data, sample_size)
    rf = RandomForestClassifier(random_state=0,
                                n_estimators=100)
    print (rf, br)
    params = {'class_weight': ['balanced'],
              'max_depth': [10, 30]}
    random = RandomizedSearchCV(rf, param_distributions = params,
                                cv=3, n_iter=2, random_state=0)
    start = time.perf_counter()
    random.fit(Xs, ys)
    see_time('RandomizedSearchCV total tuning time:')
    bp = random.best_params_
    print (bp, br)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    rf = RandomForestClassifier(**bp, random_state=0,
                                n_estimators=100)
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_scores = get_scores(rf, X_train, y_train,
                           X_test, y_test)
    see_time('total time:')
    print (rf_scores[0] + ' (train, test):')
    print (rf_scores[1], rf_scores[2], br)
    et = ExtraTreesClassifier(random_state=0, n_estimators=200)
    print (et, br)
    params = {'class_weight': ['balanced'],
              'max_depth': [10, 30]}
    random = RandomizedSearchCV(et, param_distributions = params,
                                cv=3, n_iter=2, random_state=0)
    start = time.perf_counter()
    random.fit(Xs, ys)
    see_time('RandomizedSearchCV total tuning time:')
    bp = random.best_params_
    print (bp, br)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    et = ExtraTreesClassifier(**bp, random_state=0,
                              n_estimators=200)
    start = time.perf_counter()
    et.fit(X_train, y_train)
    et_scores = get_scores(et, X_train, y_train,
                           X_test, y_test)
    see_time('total time:')
    print (et_scores[0] + ' (train, test):')
    print (et_scores[1], et_scores[2], br)
    print ('cross-validation (et):')
    start = time.perf_counter()
    scores = get_cross(rf, X, y)
    see_time('total time:')
    print (np.mean(scores), br)
    file = 'data/bp_mnist_et'
    np.save(file, bp)
    # need allow_pickle=True parameter
    bp = np.load('data/bp_mnist_et.npy', allow_pickle=True)
    bp = bp.tolist()
    print ('best parameters:')
    print (bp)
