# install humanfriendly if necessary

import numpy as np, random, humanfriendly as hf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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
    X_file = 'data/X_mnist'
    y_file = 'data/y_mnist'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    X = X.astype(np.float32)
    sample_size = 4000
    data = prep_data(X, y)
    Xs, ys = create_sample(data, sample_size, replace='no')
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.10, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train_std, X_test_std = scaler.transform(X_train),\
                              scaler.transform(X_test)
    svm = svm.SVC(random_state=0, gamma='scale')
    svm_name = svm.__class__.__name__
    print ('<<', svm_name, '>>')
    start = time.perf_counter()
    svm.fit(X_train_std, y_train)
    see_time('train:')
    start = time.perf_counter()
    y_pred = svm.predict(X_test_std)
    see_time('predict:')
    start = time.perf_counter()
    train_score = svm.score(X_train_std, y_train)
    test_score = svm.score(X_test_std, y_test)
    see_time('score:')
    print ('train score:', train_score,
           'test score', test_score, br)
    knn = KNeighborsClassifier()
    knn_name = knn.__class__.__name__
    print ('<<', knn_name, '>>')
    start = time.perf_counter()
    knn.fit(X_train, y_train)
    see_time('train:')
    start = time.perf_counter()
    y_pred = knn.predict(X_test)
    see_time('predict:')
    start = time.perf_counter()
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    see_time('score:')
    print ('train score:', train_score,
           'test score:', test_score)