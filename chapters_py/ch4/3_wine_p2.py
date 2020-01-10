import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,\
     Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt, seaborn as sns

def get_scores(model, Xtest, ytest):
    y_pred = model.predict(Xtest)
    return np.sqrt(mean_squared_error(ytest, y_pred)),\
           model.__class__.__name__

if __name__ == "__main__":
    br = '\n'
    d = dict()
    X = np.load('data/X_red.npy')
    y = np.load('data/y_red.npy')
    X_train, X_test, y_train, y_test =  train_test_split(
        X, y, test_size=0.2, random_state=0)
    print ('rmse (unscaled):')
    rfr = RandomForestRegressor(random_state=0,
                                n_estimators=100)
    rfr.fit(X_train, y_train)
    rmse, rfr_name = get_scores(rfr, X_test, y_test)
    d['rfr'] = [rmse]
    print (rmse, '(' + rfr_name + ')')
    lr = LinearRegression().fit(X_train, y_train)
    rmse, lr_name = get_scores(lr, X_test, y_test)
    d['lr'] = [rmse]
    print (rmse, '(' + lr_name + ')')
    ridge = Ridge(random_state=0).fit(X_train, y_train)
    rmse, ridge_name = get_scores(ridge, X_test, y_test)
    d['ridge'] = [rmse]
    print (rmse, '(' + ridge_name + ')')
    lasso = Lasso(random_state=0).fit(X_train, y_train)
    rmse, lasso_name = get_scores(lasso, X_test, y_test)
    d['lasso'] = [rmse]
    print (rmse, '(' + lasso_name + ')')
    en = ElasticNet(random_state=0).fit(X_train, y_train)
    rmse, en_name = get_scores(en, X_test, y_test)
    d['en'] = [rmse]
    print (rmse, '(' + en_name + ')')
    sgdr = SGDRegressor(random_state=0, max_iter=1000,
                        tol=0.001)
    sgdr.fit(X_train, y_train)
    rmse, sgdr_name = get_scores(sgdr, X_test, y_test)
    print (rmse, '(' + sgdr_name + ')', br)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)
    print ('rmse scaled:')
    lr_std = LinearRegression().fit(X_train_std, y_train)
    rmse, lr_std_name = get_scores(lr_std, X_test_std, y_test)
    print (rmse, '(' + lr_std_name + ')')
    rr_std = Ridge(random_state=0).fit(X_train_std, y_train)
    rmse, rr_std_name = get_scores(rr_std, X_test_std, y_test)
    print (rmse, '(' + rr_std_name + ')')
    lasso_std = Lasso(random_state=0).fit(X_train_std, y_train)
    rmse, lasso_std_name = get_scores(lasso_std,
                                      X_test_std, y_test)
    print (rmse, '(' + lasso_std_name + ')')
    en_std = ElasticNet(random_state=0).fit(X_train_std,
                                             y_train)
    rmse, en_std_name = get_scores(en_std,
                                   X_test_std, y_test)
    print (rmse, '(' + en_std_name + ')')
    sgdr_std = SGDRegressor(random_state=0, max_iter=1000,
                            tol=0.001)
    sgdr_std.fit(X_train_std, y_train)
    rmse, sgdr_std_name = get_scores(sgdr_std, X_test_std,
                                     y_test)
    d['sgdr_std'] = [rmse]
    print (rmse, '(' + sgdr_std_name + ')', br)
    pipe = Pipeline([('poly', PolynomialFeatures(degree=2)),
                     ('linear', LinearRegression())])
    model = pipe.fit(X_train, y_train)
    rmse, poly_name = get_scores(model, X_test, y_test)
    d['poly'] = [rmse]
    print (PolynomialFeatures().__class__.__name__, '(rmse):')
    print (rmse, '(' + poly_name + ')')
    algo, rmse = [], []
    for key, value in d.items():
        algo.append(key)
        rmse.append(value[0])
    plt.figure('RMSE')
    sns.set(style="whitegrid")
    ax = sns.barplot(algo, rmse)
    plt.title('Red Wine Algorithm Comparison')
    plt.xlabel('regressor')
    plt.ylabel('RMSE')
    plt.show()