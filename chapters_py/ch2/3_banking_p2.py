import numpy as np, pandas as pd, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,\
     ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_scores(model, xtrain, ytrain, xtest, ytest, scoring):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    f1 = f1_score(ytest, ypred, average=scoring)
    return (train, test, f1)

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

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X.npy')
    # need allow_pickle=True parameter
    y = np.load('data/y.npy', allow_pickle=True)
    print ('full data set shape for X and y:')
    print (X.shape, y.shape, br)
    X_train, X_test, y_train, y_test = train_test_split\
                                       (X, y, random_state=0)
    et = ExtraTreesClassifier(random_state=0, n_estimators=100)
    et.fit(X_train, y_train)
    et_scores = get_scores(et, X_train, y_train,
                           X_test, y_test, 'micro')
    print (et.__class__.__name__ + '(train, test, f1_score):')
    print (et_scores, br)
    rf = RandomForestClassifier(random_state=0, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_scores = get_scores(rf, X_train, y_train,
                           X_test, y_test, 'micro')
    print (rf.__class__.__name__ + '(train, test, f1_score):')
    print (rf_scores, br)
    sample_size = 4000
    data = prep_data(X, y)
    Xs, ys = create_sample(data, sample_size, replace='no')
    print ('sample data set shape for X and y:')
    print (Xs.shape, ys.shape, br)
    X_train, X_test, y_train, y_test = train_test_split\
                                       (Xs, ys, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train_std, X_test_std = scaler.transform(X_train),\
                              scaler.transform(X_test)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    knn_scores = get_scores(knn, X_train, y_train,
                            X_test, y_test, 'micro')
    print (knn.__class__.__name__ + '(train, test, f1_score):')
    print (knn_scores, br)
    svm = SVC(random_state=0, gamma='scale')
    svm.fit(X_train_std, y_train)
    svm_scores = get_scores(svm, X_train_std, y_train,
                            X_test_std, y_test, 'micro')
    print (svm.__class__.__name__ + '(train, test, f1_score):')
    print (svm_scores, br)
    knn_name, svm_name = knn.__class__.__name__,\
                         svm.__class__.__name__
    y_pred_knn = knn.predict(X_test)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_knn_T = cm_knn.T
    y_pred_svm = svm.predict(X_test_std)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_svm_T = cm_svm.T
    plt.figure(knn.__class__.__name__)
    ax = plt.axes()
    sns.heatmap(cm_knn_T, annot=True, fmt="d",
                cmap='gist_ncar_r', cbar=False)
    ax.set_title(str(knn_name) + ' confusion matrix')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.figure(str(svm_name) + ' confusion matrix' )
    ax = plt.axes()
    sns.heatmap(cm_svm_T, annot=True, fmt="d",
                cmap='gist_ncar_r', cbar=False)
    ax.set_title(svm_name)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    cnt_no, cnt_yes = 0, 0
    for i, row in enumerate(y_test):
        if row == 'no': cnt_no += 1
        elif row == 'yes': cnt_yes += 1
    cnt_no, cnt_yes = str(cnt_no), str(cnt_yes)
    print ('true =>', 'no: ' + cnt_no + ', yes: ' + cnt_yes, br)
    p_no, p_nox = cm_knn_T[0][0], cm_knn_T[0][1]
    p_yes, p_yesx = cm_knn_T[1][1], cm_knn_T[1][0]
    print ('knn classification report:')
    print ('predict \'no\':', p_no, '(' +\
           str(p_nox) + ' misclassifed)')
    print ('predict \'yes\':', p_yes, '(' +\
           str(p_yesx) + ' misclassifed)', br)
    p_no, p_nox = cm_svm_T[0][0], cm_svm_T[0][1]
    p_yes, p_yesx = cm_svm_T[1][1], cm_svm_T[1][0]
    print ('svm classification report:')
    print ('predict \'no\':', p_no, '(' +\
           str(p_nox) + ' misclassifed)')
    print ('predict \'yes\':', p_yes, '(' +\
           str(p_yesx) + ' misclassifed)')    
    plt.show()