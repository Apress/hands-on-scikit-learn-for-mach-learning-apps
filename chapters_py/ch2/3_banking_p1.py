import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    br = '\n'
    f = 'data/bank.csv'
    data = pd.read_csv(f)
    print ('original "education" categories:')
    print (data.education.unique(), br)
    data['education'] = np.where(data['education'] == 'basic.9y',
                                 'basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.6y',
                                 'basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.4y',
                                 'basic', data['education'])
    data['education'] = np.where(data['education'] ==
                                 'high.school', 'high_school',
                                 data.education)
    data['education'] = np.where(data['education'] ==
                                 'professional.course',
                                 'professional',
                                 data['education'])
    data['education'] = np.where(data['education'] ==
                                 'university.degree',
                                 'university',
                                 data['education'])
    print ('engineered "education" categories:')
    print (data.education.unique(), br)
    print ('target value counts:')
    print (data.y.value_counts(), br)
    data_X = data.loc[:, data.columns != 'y']
    cat_vars = ['job', 'marital', 'education', 'default',
                'housing', 'loan', 'contact', 'month',
                'day_of_week', 'poutcome']
    data_new = pd.get_dummies(data_X, columns=cat_vars)
    X = data_new.values
    y = data.y.values
    attributes = list(data_X)
    rf = RandomForestClassifier(random_state=0,
                                n_estimators=100)
    rf.fit(X, y)
    rf_name = rf.__class__.__name__
    feature_importances = rf.feature_importances_
    importance = sorted(zip(feature_importances, attributes),
                        reverse=True)
    n = 5
    print (n, 'most important features' + ' (' + rf_name + '):')
    [print (row) for i, row in enumerate(importance) if i < n]
    print ()
    features_file = 'data/features'
    np.save(features_file, attributes)
    features = np.load('data/features.npy')
    print ('features:')
    print (features, br)
    y_file = 'data/y'
    X_file = 'data/X'
    np.save(y_file, y)
    np.save(X_file, X)
    d = {}
    dvc = data.y.value_counts()
    d['no'], d['yes'] = dvc['no'], dvc['yes']
    dvc_file = 'data/value_counts'
    np.save(dvc_file, d)
    # need allow_pickle=True parameter
    d = np.load('data/value_counts.npy', allow_pickle=True)
    print ('class counts:', d)