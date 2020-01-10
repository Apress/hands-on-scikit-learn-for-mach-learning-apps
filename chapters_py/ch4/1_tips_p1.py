import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    br = '\n'
    tips = pd.read_csv('data/tips.csv')
    print ('original data shape:', tips.shape, br)
    target = tips['tip']
    data = tips.drop(['tip'], axis=1)
    data = pd.get_dummies(data, columns=['sex', 'smoker',
                                         'day', 'time'])
    d = {'sex_Male':'male', 'sex_Female':'female',
         'smoker_Yes':'smoker', 'smoker_No':'non-smoker',
         'day_Thur':'th', 'day_Fri':'fri', 'day_Sat':'sat',
         'day_Sun':'sun', 'time_Lunch':'lunch',
         'time_Dinner':'dinner'}
    data = data.rename(index=str, columns=d)
    X = data.values
    y = target.values
    print ('X and y shapes (post conversion):')
    print (X.shape, y.shape, br)
    X_vector = np.array([30.00, 'NaN',
                         1, 0, 1, 0, 0, 0, 0, 1, 1, 0])
    y_vector = np.array([4.5])
    X = np.vstack([X, X_vector])
    y = np.append(y, y_vector)
    print ('new X and y data point:')
    print (X[244], y[244], br)
    X_vectors = np.array([[24.99, 'NaN',
                           0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                         [19.99, 'NaN',
                          1, 0, 1, 0, 0, 0, 0, 1, 1, 0]])
    y_vectors = np.array([[3.5], [2.0]])
    X = np.vstack([X, X_vectors])
    y = np.append(y, y_vectors)
    print ('new X and y data points:')
    print (X[245], y[245])
    print (X[246], y[246], br)
    imputer = SimpleImputer()
    imputer.fit(X)
    X = imputer.transform(X)
    print ('new data shape:', X.shape, br)
    print ('new records post imputation (features and targets):')
    print (X[244], y[244])
    print (X[245], y[245])
    print (X[246], y[246], br)
    rfr = RandomForestRegressor(random_state=0, n_estimators=100)
    rfr.fit(X, y)
    print ('feature importance (first 6 features):')
    feature_importances = rfr.feature_importances_
    features = list(data.columns.values)
    importance = sorted(zip(feature_importances, features),
                        reverse=True)
    [print (row) for i, row in enumerate(importance) if i < 6]
    print ()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    model = LinearRegression()
    model_name = model.__class__.__name__
    print ('<<' + model_name + '>>', br)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print (rmse, '(rmse)', br)
    print ('predict from new data:')
    p1 = [X[244]]
    p2 = [X[245], X[246]]
    y1, y2 = model.predict(p1), model.predict(p2)
    print (y[244], y1[0])
    print (y[245], y2[0])
    print (y[246], y2[1])
    X_file = 'data/X_tips'
    y_file = 'data/y_tips'
    np.save(X_file, X)
    np.save(y_file, y)