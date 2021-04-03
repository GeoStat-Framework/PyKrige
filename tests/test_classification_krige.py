from itertools import product
import pytest

import numpy as np

from pykrige.ck import ClassificationKriging

try:
    from sklearn.svm import SVC
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.model_selection import train_test_split

    SKLEARN_INSTALLED = True
except ImportError:
    SKLEARN_INSTALLED = False


def _methods():
    krige_methods = ["ordinary", "universal"]
    ml_methods = [
        SVC(C=0.01, gamma="auto", probability=True),
        RandomForestClassifier(n_estimators=50),
    ]
    return product(ml_methods, krige_methods)


@pytest.mark.skipif(not SKLEARN_INSTALLED, reason="requires scikit-learn")
def test_classification_krige():
    np.random.seed(1)
    x = np.linspace(-1.0, 1.0, 100)
    # create a feature matrix with 5 features
    X = np.tile(x, reps=(5, 1)).T
    y = (
        1
        + 5 * X[:, 0]
        - 2 * X[:, 1]
        - 2 * X[:, 2]
        + 3 * X[:, 3]
        + 4 * X[:, 4]
        + 2 * (np.random.rand(100) - 0.5)
    )

    # create lat/lon array
    lon = np.linspace(-180.0, 180.0, 10)
    lat = np.linspace(-90.0, 90.0, 10)
    lon_lat = np.array(list(product(lon, lat)))

    discretizer = KBinsDiscretizer(encode="ordinal")
    y = discretizer.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test, lon_lat_train, lon_lat_test = train_test_split(
        X, y, lon_lat, train_size=0.7, random_state=10
    )

    for ml_model, krige_method in _methods():
        class_model = ClassificationKriging(
            classification_model=ml_model, method=krige_method, n_closest_points=2
        )
        class_model.fit(X_train, lon_lat_train, y_train)
        assert class_model.score(X_test, lon_lat_test, y_test) > 0.25


@pytest.mark.skipif(not SKLEARN_INSTALLED, reason="requires scikit-learn")
def test_krige_classification_housing():
    import ssl
    import urllib

    try:
        housing = fetch_california_housing()
    except (ssl.SSLError, urllib.error.URLError):
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            housing = fetch_california_housing()
        except PermissionError:
            # This can raise permission error on Appveyor
            pytest.skip("Failed to load california housing dataset")
        ssl._create_default_https_context = ssl.create_default_context

    # take only first 1000
    p = housing["data"][:1000, :-2]
    x = housing["data"][:1000, -2:]
    target = housing["target"][:1000]
    discretizer = KBinsDiscretizer(encode="ordinal")
    target = discretizer.fit_transform(target.reshape(-1, 1))

    p_train, p_test, y_train, y_test, x_train, x_test = train_test_split(
        p, target, x, train_size=0.7, random_state=10
    )

    for ml_model, krige_method in _methods():

        class_model = ClassificationKriging(
            classification_model=ml_model, method=krige_method, n_closest_points=2
        )
        class_model.fit(p_train, x_train, y_train)
        if krige_method == "ordinary":
            assert class_model.score(p_test, x_test, y_test) > 0.5
        else:
            assert class_model.score(p_test, x_test, y_test) > 0.0
