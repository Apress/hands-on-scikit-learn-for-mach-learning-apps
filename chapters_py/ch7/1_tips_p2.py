# install humanfriendly if necessary

import numpy as np, humanfriendly as hf
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV,\
     cross_val_score
from sklearn.metrics import mean_squared_error

def get_error(model, Xtest, ytest):
    y_pred = model.predict(Xtest)
    return np.sqrt(mean_squared_error(ytest, y_pred)),\
           model.__class__.__name__

def see_time(note):
    end = time.perf_counter()
    elapsed = end - start
    print (note,
           hf.format_timespan(elapsed, detailed=True))

def get_cross(model, data, target, groups=10):
    return cross_val_score(model, data, target, cv=groups,
                           scoring='neg_mean_squared_error')

if __name__ == "__main__":
    br = '\n'
    X = np.load('data/X_tips.npy')
    y = np.load('data/y_tips.npy')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)
    lasso = Lasso(random_state=0, alpha=0.1)
    print (lasso, br)
    lasso.fit(X_train_std, y_train)
    rmse, name = get_error(lasso, X_test_std, y_test)
    print (name + '(rmse):', end=' ')
    print (rmse, br)
    alpha_lasso = [1e-1]
    params = {'alpha': alpha_lasso, 'positive': [True, False],
              'max_iter': [10, 50, 100]}
    grid = GridSearchCV(lasso, params, cv=5, n_jobs=-1,
                        verbose=1)
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    see_time('training time:')
    bp = grid.best_params_
    print (bp, br)
    lasso = Lasso(**bp, random_state=0).fit(X_train_std, y_train)
    rmse, name = get_error(lasso, X_test_std, y_test)
    print (name + '(rmse):', end=' ')
    print (rmse, br)
    start = time.perf_counter()
    scores = get_cross(lasso, X, y)
    see_time('cross-validation rmse:')
    rmse = np.sqrt(np.mean(scores) * -1)
    print (rmse)