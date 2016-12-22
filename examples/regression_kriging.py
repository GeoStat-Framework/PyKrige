# coding: utf-8
import numpy as np
from itertools import product
from pykrige.rk import RegressionKriging
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

svr_model = SVR(C=0.1)
rf_model = RandomForestRegressor(n_estimators=100)
lr_model = LinearRegression()

models = [svr_model, rf_model, lr_model]

# dummy data

np.random.seed(10)
X = np.random.rand(100, 5)
y = 2 + 10*X[:, 0] - 10*X[:, 1] - X[:, 2] + 3*X[:, 3] + 4*X[:, 4]
lon = np.linspace(-180., 180.0, 20)
lat = np.linspace(-90., 90., 20)
np.random.shuffle(lon)
np.random.shuffle(lat)

lon_lat = np.array(list(product(lon[:10], lat[:10])))

for m in models:
    m_rk = RegressionKriging(ml_model=m, n_closest_points=15)
    m_rk.fit(X[:70, :], lon_lat[:70, :], y[:70])
    print('='*40, '\n', 'regression model:', m)
    print(m_rk.predict(X[70:, :], lon_lat[70:, :]))
    print('score: ', m_rk.score(X[70:, :], lon_lat[70:, :], y[70:]))
