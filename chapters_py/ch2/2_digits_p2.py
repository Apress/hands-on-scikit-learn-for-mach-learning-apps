# install humanfriendly if necessary

import humanfriendly as hf
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,\
     LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,\
     ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def get_scores(model, Xtest, ytest, avg):
    y_pred = model.predict(Xtest)
    accuracy = accuracy_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred, average=avg)
    return (accuracy, f1)

def get_time(time):
    return hf.format_timespan(time, detailed=True)

if __name__ == "__main__":
    br = '\n'
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split\
                                       (X, y, random_state=0)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    lr = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='auto', max_iter=4000)
    lr.fit(X_train_std, y_train)
    lr_name = lr.__class__.__name__
    acc, f1 = get_scores(lr, X_test_std, y_test, 'micro')
    print (lr_name + ' scaled \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    softmax = LogisticRegression(multi_class="multinomial",
                                 solver="lbfgs", max_iter=4000,
                                 C=10, random_state=0)
    softmax.fit(X_train_std, y_train)
    acc, f1 = get_scores(softmax, X_test_std, y_test, 'micro')
    print (lr_name + ' (softmax) scaled \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    rf = RandomForestClassifier(random_state=0,
                                n_estimators=100)
    rf.fit(X_train_std, y_train)
    rf_name = rf.__class__.__name__
    acc, f1 = get_scores(rf, X_test_std, y_test, 'micro')
    print (rf_name + ' \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    et = ExtraTreesClassifier(random_state=0,
                              n_estimators=100)
    et.fit(X_train, y_train)
    et_name = et.__class__.__name__
    acc, f1 = get_scores(et, X_test, y_test, 'micro')
    print (et_name + ' \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    gboost_clf = GradientBoostingClassifier(random_state=0)
    gb_name = gboost_clf.__class__.__name__
    gboost_clf.fit(X_train, y_train)
    acc, f1 = get_scores(gboost_clf, X_test, y_test, 'micro')
    print (gb_name + ' \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    knn_clf = KNeighborsClassifier().fit(X_train, y_train)
    knn_name = knn_clf.__class__.__name__
    acc, f1 = get_scores(knn_clf, X_test, y_test, 'micro')
    print (knn_name + ' \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1, br)
    start = time.perf_counter()
    lr_cv = LogisticRegressionCV(random_state=0, cv=5,
                                 multi_class='auto',
                                 max_iter=4000)
    lr_cv_name = lr_cv.__class__.__name__
    lr_cv.fit(X, y)
    end = time.perf_counter()
    elapsed_ls = end - start
    timer = get_time(elapsed_ls)
    print (lr_cv_name + ' timer:', timer)
    acc, f1 = get_scores(lr_cv, X_test, y_test, 'micro')
    print (lr_cv_name + ' \'test\':')
    print ('accuracy:', acc, ', f1_score:', f1)