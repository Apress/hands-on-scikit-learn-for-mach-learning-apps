# install humanfriendly if necessary

import numpy as np, humanfriendly as hf, random
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def get_scores(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    name = model.__class__.__name__
    return (name, train, test)

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
    bp = np.load('data/bp_bank.npy', allow_pickle=True)
    bp = bp.tolist()
    print ('best parameters:')
    print (bp, br)
    X_train, X_test, y_train, y_test = train_test_split\
                                       (X, y, random_state=0)
    start = time.perf_counter()
    knn = KNeighborsClassifier(**bp)
    knn.fit(X_train, y_train)
    see_time('training time:')
    start = time.perf_counter()
    knn_scores = get_scores(knn, X_train, y_train,
                            X_test, y_test)
    see_time('scoring time:')
    print ()
    print (knn_scores[0] + ' (train, test):')
    print (knn_scores[1], knn_scores[2])