from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_scores(model, Xtrain, Xtest, ytrain, ytest):
    y_ptrain = model.predict(Xtrain)
    y_ptest = model.predict(Xtest)
    acc_train = accuracy_score(ytrain, y_ptrain)
    acc_test = accuracy_score(ytest, y_ptest)
    name = model.__class__.__name__
    return (name, acc_train, acc_test)

if __name__ == "__main__":
    br = '\n'
    X, y = datasets.make_moons(n_samples=1000, shuffle=True,
                               noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    accuracy = get_scores(knn, X_train, X_test, y_train, y_test)
    print ('<<' + str(accuracy[0]) + '>>')
    print ('train:', accuracy[1], 'test:', accuracy[2], br)
    svm = svm.SVC(gamma='scale', random_state=0)
    svm.fit(X_train, y_train)
    accuracy = get_scores(svm, X_train, X_test, y_train, y_test)
    print ('<<' + str(accuracy[0]) + '>>')
    print ('train:', accuracy[1], 'test:', accuracy[2])