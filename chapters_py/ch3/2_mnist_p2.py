# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

if __name__ == "__main__":
    br = '\n'
    X_file = 'data/X_mnist'
    y_file = 'data/y_mnist'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:],\
                                       y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index],\
                       y_train[shuffle_index]
    et = ExtraTreesClassifier(random_state=0, n_estimators=100)
    start = time.perf_counter()
    et.fit(X_train, y_train)
    end = time.perf_counter()
    elapsed_ls = end - start
    print (hf.format_timespan(elapsed_ls, detailed=True))
    et_name = et.__class__.__name__
    y_pred = et.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print (et_name + ' \'test\':', end=' ')
    print ('accuracy:', accuracy, br)
    rpt = classification_report(y_test, y_pred)
    print (rpt)