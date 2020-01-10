# install humanfriendly if necessary

import numpy as np, random, humanfriendly as hf
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

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

def get_scores(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    f1 = f1_score(ytest, ypred, average='macro')
    return (ypred, train, test, f1)

if __name__ == "__main__":
    br = '\n'
    X_file = 'data/X_mnist'
    y_file = 'data/y_mnist'
    X = np.load('data/X_mnist.npy')
    y = np.load('data/y_mnist.npy')
    X = X.astype(np.float32)
    data = prep_data(X, y)
    sample_size = 7000
    Xs, ys = create_sample(data, sample_size, replace='no')
    pca = PCA(n_components=0.95, random_state=0)
    Xs_reduced = pca.fit_transform(Xs)
    print ('sample feature shape:', Xs.shape)
    components = pca.n_components_
    print ('feature components with PCA:', components, br)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs_reduced, ys, test_size=0.10, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train_std, X_test_std = scaler.transform(X_train),\
                              scaler.transform(X_test)
    start = time.perf_counter()
    svm = svm.SVC(random_state=0).fit(X_train_std, y_train)
    svm_name = svm.__class__.__name__
    svm_scores = get_scores(svm, X_train_std, y_train,
                            X_test_std, y_test)
    cm_svm = confusion_matrix(y_test, svm_scores[0])
    see_time(svm_name + ' total training time:')
    print (svm_name + ':', svm_scores[1], svm_scores[2],
           svm_scores[3], br)
    start = time.perf_counter()
    knn = KNeighborsClassifier().fit(X_train, y_train)
    knn_name = knn.__class__.__name__
    knn_scores = get_scores(knn, X_train, y_train,
                            X_test, y_test)
    cm_knn = confusion_matrix(y_test, knn_scores[0])
    see_time(knn_name + ' total training time:')
    print (knn_name + ':', knn_scores[1], knn_scores[2],
           knn_scores[3])
    plt.figure(svm_name)
    ax = plt.axes()
    sns.heatmap(cm_svm.T, annot=True, fmt="d",
                cmap='gist_ncar_r', ax=ax)
    ax.set_title(str(svm_name) + ' confustion matrix')
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.figure(knn_name)
    ax = plt.axes()
    sns.heatmap(cm_knn.T, annot=True, fmt="d",
                cmap='gist_ncar_r', ax=ax)
    ax.set_title(str(knn_name) + ' confustion matrix')
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.show()