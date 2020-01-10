from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import\
     LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import *

if __name__ == "__main__":
    br = '\n'
    data = load_wine()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0)
    lda = LDA().fit(X_train, y_train)
    print (lda, br)
    lda_name = lda.__class__.__name__
    y_pred = lda.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    accuracy = str(accuracy * 100) + '%'
    print (lda_name + ':')
    print ('train:', accuracy)
    y_pred_test = lda.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_test)
    accuracy = str(round(accuracy * 100, 2)) + '%'
    print ('test: ', accuracy, br)
    print('Confusion Matrix', lda_name)
    print(metrics.confusion_matrix(y_test,
                                   lda.predict(X_test)), br)
    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)
    sgd = SGDClassifier(max_iter=5, random_state=0)
    print (sgd, br)
    sgd.fit(X_train, y_train)
    sgd_name = sgd.__class__.__name__
    y_pred = sgd.predict(X_train)
    y_pred_test = sgd.predict(X_test)
    print (sgd_name + ':')
    print('train: {:.2%}'.format(metrics.accuracy_score\
                                 (y_train, y_pred)))
    print('test:  {:.2%}\n'.format(metrics.accuracy_score\
                                   (y_test, y_pred_test)))
    print('Confusion Matrix', sgd_name)
    print(metrics.confusion_matrix(y_test,
                                   sgd.predict(X_test)), br)
    n, ls = 100, []
    for i, row in enumerate(range(n)):
        rs = randint(0, 100)
        sgd = SGDClassifier(max_iter=5, random_state=0)
        sgd.fit(X_train, y_train)
        y_pred = sgd.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls.append(accuracy)
    avg = sum(ls) / len(ls)
    print ('MCS (true test accuracy):', avg)