import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,\
     ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def get_scores(model, xtrain, ytrain, xtest, ytest, scoring):
    ypred = model.predict(xtest)
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, y_test)
    f1 = f1_score(ytest, ypred, average=scoring)
    return (train, test, f1)

if __name__ == "__main__":
    br = '\n'
    f = 'data/bank_sample.csv'
    data = pd.read_csv(f)
    print ('data shape:', data.shape, br)
    data['education'] =\
                      np.where(data['education'] == 'basic.9y',
                               'basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.6y',
                                 'basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.4y',
                                 'basic', data['education'])
    data['education'] = np.where(data['education'] ==
                                 'high.school',
                                 'high_school',
                                 data.education)
    data['education'] = np.where(data['education'] ==
                                 'professional.course',
                                 'professional',
                                 data['education'])
    data['education'] = np.where(data['education'] ==
                                 'university.degree',
                                 'university',
                                 data['education'])
    data_X = data.loc[:, data.columns != 'y']
    cat_vars = ['job', 'marital', 'education', 'default',
                'housing', 'loan', 'contact', 'month',
                'day_of_week', 'poutcome']
    data_new = pd.get_dummies(data_X, columns=cat_vars)
    attributes = list(data_X)
    y = data.y.values
    X = data_new.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    rf = RandomForestClassifier(random_state=0,
                                n_estimators=100)
    rf.fit(X_train, y_train)
    rf_name = rf.__class__.__name__
    rf_scores = get_scores(rf, X_train, y_train,
                           X_test, y_test, 'micro')
    print (rf.__class__.__name__ + '(train, test, f1_score):')
    print (rf_scores, br)
    et = ExtraTreesClassifier(random_state=0, n_estimators=100)
    et.fit(X_train, y_train)
    et_name = et.__class__.__name__
    et_scores = get_scores(et, X_train, y_train,
                           X_test, y_test, 'micro')
    print (et.__class__.__name__ + '(train, test, f1_score):')
    print (et_scores, br)
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
    print (svm_scores)