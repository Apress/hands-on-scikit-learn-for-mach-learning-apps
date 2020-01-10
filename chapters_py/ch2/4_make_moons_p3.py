from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_scores(model, Xtrain, ytrain, Xtest, ytest,
               Xvalid, yvalid):
    y_ptrain = model.predict(Xtrain)
    y_ptest = model.predict(Xtest)
    y_pvalid = model.predict(Xvalid)
    acc_train = accuracy_score(ytrain, y_ptrain)
    acc_test = accuracy_score(ytest, y_ptest)
    acc_valid = accuracy_score(yvalid, y_pvalid)
    name = model.__class__.__name__
    return (name, acc_train, acc_test, acc_valid)

if __name__ == "__main__":
    br = '\n'
    X_train, y_train = make_moons(n_samples=1000, shuffle=True,
                                  noise=0.2, random_state=0)
    X_test, y_test = make_moons(n_samples=1000, shuffle=True,
                                noise=0.2, random_state=0)
    X_valid, y_valid = make_moons(n_samples=10000, shuffle=True,
                                  noise=0.2, random_state=0)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    accuracy = get_scores(knn, X_train, y_train, X_test, y_test,
                          X_valid, y_valid)
    print ('train test valid split (technique 1):')
    print ('<<' + str(accuracy[0]) + '>>')
    print ('train:', accuracy[1], 'test:', accuracy[2],
           'valid:', accuracy[3])
    print ('sample split:', X_train.shape, X_test.shape,
           X_valid.shape)
    print ()
    X, y = make_moons(n_samples=1000, shuffle=True, noise=0.2,
                               random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    accuracy = get_scores(knn, X_train, y_train, X_test, y_test,
                          X_valid, y_valid)
    print ('train test valid split (technique 2):')
    print ('<<' + str(accuracy[0]) + '>>')
    print ('train:', accuracy[1], 'test:', accuracy[2],
           'valid:', accuracy[3])
    print ('sample split:', X_train.shape, X_test.shape,
           X_val.shape)