# coding: utf-8
"""Classification Kriging."""
import numpy as np
from pykrige.compat import Krige, validate_sklearn, check_sklearn_model

validate_sklearn()

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from scipy.linalg import helmert


class ClassificationKriging:
    """
    An implementation of Simplicial Indicator Kriging applied to classification ilr transformed residuals.

    Parameters
    ----------
    classification_model: machine learning model instance from sklearn
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
        classification_model=SVC(),
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
        check_sklearn_model(classification_model, task="classification")
        self.classification_model = classification_model
        self.n_closest_points = n_closest_points
        self._kriging_kwargs = dict(
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
        Fit the classification method and also krige the residual.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for classification
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example 2d classification kriging.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (Ns, )
        """
        self.classification_model.fit(p, y.ravel())
        print("Finished learning classification model")
        self.classes_ = self.classification_model.classes_

        self.krige = []
        for i in range(len(self.classes_) - 1):
            self.krige.append(Krige(**self._kriging_kwargs))

        ml_pred = self.classification_model.predict_proba(p)
        ml_pred_ilr = ilr_transformation(ml_pred)

        self.onehotencode = OneHotEncoder(categories=[self.classes_])
        y_ohe = np.array(self.onehotencode.fit_transform(y).todense())
        y_ohe_ilr = ilr_transformation(y_ohe)

        for i in range(len(self.classes_) - 1):
            self.krige[i].fit(x=x, y=y_ohe_ilr[:, i] - ml_pred_ilr[:, i])

        print("Finished kriging residuals")

    def predict(self, p, x, **kwargs):
        """
        Predict.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for classification
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging

        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, of shape (Ns,).

        """

        ml_pred = self.classification_model.predict_proba(p)
        ml_pred_ilr = ilr_transformation(ml_pred)

        pred_proba_ilr = self.krige_residual(x, **kwargs) + ml_pred_ilr
        pred_proba = inverse_ilr_transformation(pred_proba_ilr)

        return np.argmax(pred_proba, axis=1)

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

        krig_pred = [
            self.krige[i].predict(x=x, **kwargs) for i in range(len(self.classes_) - 1)
        ]

        return np.vstack(krig_pred).T

    def score(self, p, x, y, sample_weight=None, **kwargs):
        """
        Overloading default classification score method.

        Parameters
        ----------
        p: ndarray
            (Ns, d) array of predictor variables (Ns samples, d dimensions)
            for classification
        x: ndarray
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (Ns, )
        """
        return accuracy_score(
            y_pred=self.predict(p, x, **kwargs), y_true=y, sample_weight=sample_weight
        )


def closure(data, k=1.0):
    """Apply closure to data, sample-wise.
    Adapted from https://github.com/ofgulban/compoda.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Data to be closed to a certain constant. Do not forget to deal with
        zeros in the data before this operation.
    k : float, positive
        Sum of the measurements will be equal to this number.

    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 9.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144
    """

    return k * data / np.sum(data, axis=1)[:, np.newaxis]


def ilr_transformation(data):
    """Isometric logratio transformation (not vectorized).
    Adapted from https://github.com/ofgulban/compoda.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Barycentric coordinates (closed) in simplex space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates-1]
        Coordinates in real space.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 37.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144
    """
    data = np.maximum(data, np.finfo(float).eps)

    return np.einsum("ij,jk->ik", np.log(data), -helmert(data.shape[1]).T)


def inverse_ilr_transformation(data):
    """Inverse isometric logratio transformation (not vectorized).
    Adapted from https://github.com/ofgulban/compoda.

    Parameters
    ----------
    data : 2d numpy array, shape [n_samples, n_coordinates]
        Isometric log-ratio transformed coordinates in real space.

    Returns
    -------
    out : 2d numpy array, shape [n_samples, n_coordinates+1]
        Barycentric coordinates (closed) in simplex space.

    Reference
    ---------
    [1] Pawlowsky-Glahn, V., Egozcue, J. J., & Tolosana-Delgado, R.
        (2015). Modelling and Analysis of Compositional Data, pg. 37.
        Chichester, UK: John Wiley & Sons, Ltd.
        DOI: 10.1002/9781119003144
    """

    return closure(np.exp(np.einsum("ij,jk->ik", data, -helmert(data.shape[1] + 1))))
