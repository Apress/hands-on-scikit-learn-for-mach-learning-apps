import numpy as np
from sklearn.linear_model import LinearRegression, Ridge,\
     Lasso, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def get_scores(model, Xtest, ytest):
    y_pred = model.predict(Xtest)
    return np.sqrt(mean_squared_error(ytest, y_pred)),\
           model.__class__.__name__

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_tips.npy')
    y = np.load('data/y_tips.npy')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    print ('rmse:')
    lr = LinearRegression().fit(X_train, y_train)
    rmse, lr_name = get_scores(lr, X_test, y_test)
    print (rmse, '(' + lr_name + ')')
    rr = Ridge(random_state=0).fit(X_train, y_train)
    rmse, rr_name = get_scores(rr, X_test, y_test)
    print (rmse, '(' + rr_name + ')')
    lasso = Lasso(random_state=0).fit(X_train, y_train)
    rmse, lasso_name = get_scores(lasso, X_test, y_test)
    print (rmse, '(' + lasso_name + ')')
    en = ElasticNet(random_state=0).fit(X_train, y_train)
    rmse, en_name = get_scores(en, X_test, y_test)
    print (rmse, '(' + en_name + ')')
    sgdr = SGDRegressor(random_state=0,
                        max_iter=1000, tol=0.001)
    sgdr.fit(X_train, y_train)
    rmse, sgdr_name = get_scores(sgdr, X_test, y_test)
    print (rmse, '(' + sgdr_name + ')', br)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)
    print ('rmse std:')
    lr_std = LinearRegression().fit(X_train_std, y_train)
    rmse, lr_name = get_scores(lr_std, X_test_std, y_test)
    print (rmse, '(' + lr_name + ')')
    rr_std = Ridge(random_state=0).fit(X_train_std, y_train)
    rmse, rr_name = get_scores(rr_std, X_test_std, y_test)
    print (rmse, '(' + rr_name + ')')
    lasso_std = Lasso(random_state=0).fit(X_train_std, y_train)
    rmse, lasso_name = get_scores(lasso_std, X_test_std, y_test)
    print (rmse, '(' + lasso_name + ')')
    en_std = ElasticNet(random_state=0)
    en_std.fit(X_train_std, y_train)
    rmse, en_name = get_scores(en_std, X_test_std, y_test)
    print (rmse, '(' + en_name + ')')
    sgdr_std = SGDRegressor(random_state=0,
                            max_iter=1000, tol=0.001)
    sgdr_std.fit(X_train_std, y_train)
    rmse, sgdr_name = get_scores(sgdr_std, X_test_std, y_test)
    print (rmse, '(' + sgdr_name + ')')