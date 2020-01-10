# install humanfriendly if necessary

import numpy as np, humanfriendly as hf, random
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_scores(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    name = model.__class__.__name__
    return (name, train, test)

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
    pca = PCA(n_components=0.95, random_state=0)
    Xs = StandardScaler().fit_transform(Xs)
    Xs_reduced = pca.fit_transform(Xs)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs_reduced, ys, random_state=0)
    svm = SVC(gamma='scale', random_state=0)
    print (svm, br)
    start = time.perf_counter()
    svm.fit(X_train, y_train)
    svm_scores = get_scores(svm, X_train, y_train,
                            X_test, y_test)
    print (svm_scores[0] + ' (train, test):')
    print (svm_scores[1], svm_scores[2])
    see_time('time:')
    print ()
    param_grid = {'C': [30, 35, 40], 'kernel': ['poly'],
                  'gamma': ['scale'], 'degree': [3],
                  'coef0': [0.1]}
    start = time.perf_counter()
    rand = RandomizedSearchCV(svm, param_grid, cv=3, n_jobs = -1,
                              random_state=0, n_iter=3,
                              verbose=2)
    rand.fit(X_train, y_train)
    see_time('RandomizedSearchCV total tuning time:')
    bp = rand.best_params_
    print (bp, br)
    svm = SVC(**bp, random_state=0)
    start = time.perf_counter()
    svm.fit(X_train, y_train)
    svm_scores = get_scores(svm, X_train, y_train,
                            X_test, y_test)
    print (svm_scores[0] + ' (train, test):')
    print (svm_scores[1], svm_scores[2])
    see_time('total time:')