# coding: utf-8
"""Regression Kriging."""
from pykrige.compat import Krige, validate_sklearn, check_sklearn_model

validate_sklearn()

from sklearn.metrics import r2_score
from sklearn.svm import SVR


class RegressionKriging:
    """
    An implementation of Regression-Kriging.

    As described here:
    https://en.wikipedia.org/wiki/Regression-Kriging

    Parameters
    ----------
    regression_model: machine learning model instance from sklearn
    method: str, optional
        type of kriging to be performed
    variogram_model: str, optional
        variogram model to be used during Kriging
    n_closest_points: int
        number of closest points to be used during Ordinary Kriging
    nlags: int
        see OK/UK class description
    weight: bool
        see OK/UK class description
    verbose: bool
        see OK/UK class description
    exact_values : bool
        see OK/UK class description
    variogram_parameters : list or dict
        see OK/UK class description
    variogram_function : callable
        see OK/UK class description
    anisotropy_scaling : tuple
        single value for 2D (UK/OK) and two values in 3D (UK3D/OK3D)
    anisotropy_angle : tuple
        single value for 2D (UK/OK) and three values in 3D (UK3D/OK3D)
    enable_statistics : bool
        see OK class description
    coordinates_type : str
        see OK/UK class description
    drift_terms : list of strings
        see UK/UK3D class description
    point_drift : array_like
        see UK class description
    ext_drift_grid : tuple
        Holding the three values external_drift, external_drift_x and
        external_drift_z for the UK class
    functional_drift : list of callable
        see UK/UK3D class description
    """

    def __init__(
        self,
        regression_model=SVR(),
        method="ordinary",
        variogram_model="linear",
        n_closest_points=10,
        nlags=6,
        weight=False,
        verbose=False,
        exact_values=True,
        pseudo_inv=False,
        pseudo_inv_type="pinv",
        variogram_parameters=None,
        variogram_function=None,
        anisotropy_scaling=(1.0, 1.0),
        anisotropy_angle=(0.0, 0.0, 0.0),
        enable_statistics=False,
        coordinates_type="euclidean",
        drift_terms=None,
        point_drift=None,
        ext_drift_grid=(None, None, None),
        functional_drift=None,
    ):
        check_sklearn_model(regression_model)
        self.regression_model = regression_model
        self.n_closest_points = n_closest_points
        self.krige = Krige(
            method=method,
            variogram_model=variogram_model,
            nlags=nlags,
            weight=weight,
            n_closest_points=n_closest_points,
            verbose=verbose,
            exact_values=exact_values,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            variogram_parameters=variogram_parameters,
            variogram_function=variogram_function,
            anisotropy_scaling=anisotropy_scaling,
            anisotropy_angle=anisotropy_angle,
            enable_statistics=enable_statistics,
            coordinates_type=coordinates_type,
            drift_terms=drift_terms,
            point_drift=point_drift,
            ext_drift_grid=ext_drift_grid,
            functional_drift=functional_drift,
        )

    def fit(self, p, x, y):
        """
        Fit the regression method and also Krige the residual.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for regression
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example 2d regression kriging.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (Ns, )
        """
        self.regression_model.fit(p, y)
        ml_pred = self.regression_model.predict(p)
        print("Finished learning regression model")
        # residual=y-ml_pred
        self.krige.fit(x=x, y=y - ml_pred)
        print("Finished kriging residuals")

    def predict(self, p, x, **kwargs):
        """
        Predict.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for regression
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging

        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, of shape (Ns,).

        """
        return self.krige_residual(x, **kwargs) + self.regression_model.predict(p)

    def krige_residual(self, x, **kwargs):
        """
        Calculate the residuals.

        Parameters
        ----------
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.

        Returns
        -------
        residual: ndarray
            kriged residual values
        """
        return self.krige.predict(x, **kwargs)

    def score(self, p, x, y, sample_weight=None, **kwargs):
        """
        Overloading default regression score method.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for regression
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (Ns, )
        """
        return r2_score(
            y_pred=self.predict(p, x, **kwargs), y_true=y, sample_weight=sample_weight
        )
