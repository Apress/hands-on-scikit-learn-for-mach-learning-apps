import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

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
    poly = PolynomialFeatures(degree=2)
    poly.fit(X_train, y_train)
    X_train_poly = poly.transform(X_train)
    lr = LinearRegression().fit(X_train_poly, y_train)
    X_test_poly = poly.transform(X_test)
    rmse, lr_name = get_scores(lr, X_test_poly, y_test)
    print (rmse, '(squared polynomial fitting)')
    poly = PolynomialFeatures(degree=3)
    poly.fit(X_train, y_train)
    X_train_poly = poly.transform(X_train)
    lr = LinearRegression().fit(X_train_poly, y_train)
    X_test_poly = poly.transform(X_test)
    rmse, lr_name = get_scores(lr, X_test_poly, y_test)
    print (rmse, '(cubic polynomial fitting)')
    poly = PolynomialFeatures(degree=4)
    poly.fit(X_train, y_train)
    X_train_poly = poly.transform(X_train)
    lr = LinearRegression().fit(X_train_poly, y_train)
    X_test_poly = poly.transform(X_test)
    rmse, lr_name = get_scores(lr, X_test_poly, y_test)
    print (rmse, '(quartic polynomial fitting)')