from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.gaussian_process import\
     GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF as rbf
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf,\
     AdaBoostClassifier as ada
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.discriminant_analysis import\
     QuadraticDiscriminantAnalysis as qda,\
     LinearDiscriminantAnalysis as lda
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == "__main__":
    br = '\n'
    wine = load_wine()
    X = wine.data
    y = wine.target
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=0)
    classifiers = [knn(3), qda(), lda(), gnb(), 
                   SVC(kernel='linear', gamma='scale',
                       random_state=0),
                   ada(random_state=0), dt(random_state=0),
                   sgd(max_iter=100, random_state=0),
                   gpc(1.0 * rbf(1.0), random_state=0),
                   rf(random_state=0, n_estimators=100)]
    for clf in classifiers:
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        name = clf.__class__.__name__
        print (name + '(train/test scores):')
        print (train_score, test_score)