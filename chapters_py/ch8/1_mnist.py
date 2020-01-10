# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_scores(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    name = model.__class__.__name__
    return (name, train, test, ypred)

def see_time(note):
    end = time.perf_counter()
    elapsed = end - start
    print (note,
           hf.format_timespan(elapsed, detailed=True))

def find_misses(test, pred):
    return [i for i, row in enumerate(test) if row != pred[i]]

if __name__ == "__main__":
    br = '\n'
    X_file = 'data/X_mnist'
    y_file = 'data/y_mnist'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    X = X.astype(np.float32)
    # need allow_pickle=True parameter
    bp = np.load('data/bp_mnist_et.npy', allow_pickle=True)
    bp = bp.tolist()
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
    y_pred = et_scores[3]
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(1)
    ax = plt.axes()
    sns.heatmap(cm.T, annot=True, fmt="d",
                cmap='gist_ncar_r', ax=ax)
    ax.set_title(et_scores[0] + 'confustion matrix')
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    indx = find_misses(y_test, y_pred)
    print ('pred', 'actual')
    misses = [(y_pred[row], y_test[row], i)
              for i, row in enumerate(indx)]
    [print (row[0], '  ', row[1]) for i, row in enumerate(misses)
     if i < 10]
    print()
    img_act = y_test[indx[0]]
    img_pred = y_pred[indx[0]]
    print ('actual', img_act)
    print ('pred', img_pred)
    text = str(img_pred)
    test_images = X_test.reshape(-1, 28, 28)
    plt.figure(2)
    plt.imshow(test_images[indx[0]], cmap='gray',
               interpolation='gaussian')
    plt.text(0, 0.05, text, color='r',
             bbox=dict(facecolor='white'))
    title = str(img_act) + ' misclassified as ' + text
    plt.title(title)
    plt.show()