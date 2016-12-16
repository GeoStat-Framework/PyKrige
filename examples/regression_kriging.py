import numpy as np
from pykrige.compat import SKLEARN_INSTALLED, validate_sklearn

if SKLEARN_INSTALLED:
    from pykrige.rk import MLKrige
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

validate_sklearn()

svr_model = SVR(C=0.1)
rf_model = RandomForestRegressor(n_estimators=100)
lr_model = LinearRegression()

models = [svr_model, rf_model, lr_model]

# dummy data
X = np.random.randint(0, 400, size=(100, 10)).astype(float)
y = 5 * np.random.rand(100)
lon_lat = np.random.randint(0, 400, size=(100, 2)).astype(float)

for m in models:
    m_rk = MLKrige(ml_model=svr_model)
    m_rk.fit(X[:70, :], lon_lat[:70, :], y[:70])
    print('='*40, '\n', 'regression model:', m)
    print(m_rk.predict(X[70:, :], lon_lat[70:, :]))
    print('score: ', m_rk.score(X[70:, :], lon_lat[70:, :], y[70:]))
