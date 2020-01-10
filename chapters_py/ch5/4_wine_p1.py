import numpy as np, random
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis\
     import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,\
     cross_val_score
from sklearn import metrics

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups)

if __name__ == "__main__":
    br = '\n'
    wine = load_wine()
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
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
    print ('cross-validation:')
    scores = get_cross(lda, X, y)
    print (np.mean(scores), br)
    n, ls = 100, []
    for i, row in enumerate(range(n)):
        rs = random.randint(1, 100)
        sgd = LDA().fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls.append(accuracy)
    avg = sum(ls) / len(ls)
    print ('MCS')
    print (avg, br)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    sgd = SGDClassifier(max_iter=100, random_state=1)
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
    print ('cross-validation:')
    scores = get_cross(sgd, X, y)
    print (np.mean(scores), br)
    n, ls = 100, []
    for i, row in enumerate(range(n)):
        rs = random.randint(1, 100)
        sgd = SGDClassifier(max_iter=100).fit(X_train, y_train)
        y_pred = sgd.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        ls.append(accuracy)
    avg = sum(ls) / len(ls)
    print ('MCS:')
    print (avg)