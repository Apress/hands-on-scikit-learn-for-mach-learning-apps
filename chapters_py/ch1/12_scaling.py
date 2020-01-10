import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    br = '\n'
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test =\
             train_test_split(X, y, random_state=0)
    sgd = SGDClassifier(random_state=0, max_iter=1000,
                        tol=0.001)
    sgd.fit(X_train, y_train)
    sgd_name = sgd.__class__.__name__
    print ('<<' + sgd_name + '>>', br)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ('unscaled \'test\' accuracy:', accuracy)
    scaler = StandardScaler().fit(X_train)
    X_train_std, X_test_std = scaler.transform(X_train),\
                              scaler.transform(X_test)
    sgd_std = SGDClassifier(random_state=0, max_iter=1000,
                            tol=0.001)
    sgd_std.fit(X_train_std, y_train)
    y_pred = sgd_std.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    print ('scaled \'test\' accuracy:', np.round(accuracy, 4))