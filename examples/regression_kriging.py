# coding: utf-8
from pykrige.rk import RegressionKriging
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

svr_model = SVR(C=0.1)
rf_model = RandomForestRegressor(n_estimators=100)
lr_model = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)

models = [svr_model, rf_model, lr_model]

# take the first 5000 as Kriging is memory intensive
housing = fetch_california_housing()
X = housing['data'][:5000, :-2]
lat_lon = housing['data'][:5000, -2:]
target = housing['target'][:5000]

X_train, X_test, lat_lon_train, lat_lon_test, target_train, target_test \
    = train_test_split(X, lat_lon, target, test_size=0.3, random_state=42)

for m in models:
    print('=' * 40, '\n', 'regression model:', m.__class__)
    m_rk = RegressionKriging(ml_model=m, n_closest_points=10)
    m_rk.fit(X_train, lat_lon_train, target_train)
    print('Regression Score: ', m_rk.ml_model.score(X_test, target_test))
    print('RK score: ', m_rk.score(X_test, lat_lon_test, target_test))

##====================================OUTPUT==================================

# ========================================
#  regression model: <class 'sklearn.svm.classes.SVR'>
# Finished learning regression model
# Finished kriging residuals
# Regression Score:  -0.034053855457
# RK score:  0.66195576665
# ========================================
#  regression model: <class 'sklearn.ensemble.forest.RandomForestRegressor'>
# Finished learning regression model
# Finished kriging residuals
# Regression Score:  0.699771164651
# RK score:  0.737574040386
# ========================================
#  regression model: <class 'sklearn.linear_model.base.LinearRegression'>
# Finished learning regression model
# Finished kriging residuals
# Regression Score:  0.527796839838
# RK score:  0.604908933617
