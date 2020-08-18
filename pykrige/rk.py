# coding: utf-8
from pykrige.compat import validate_sklearn

validate_sklearn()
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.svm import SVR
from sklearn.metrics import r2_score

krige_methods = {
    "ordinary": OrdinaryKriging,
    "universal": UniversalKriging,
    "ordinary3d": OrdinaryKriging3D,
    "universal3d": UniversalKriging3D,
}

threed_krige = ("ordinary3d", "universal3d")

krige_methods_kws = {
    "ordinary": [
        "anisotropy_scaling",
        "anisotropy_angle",
        "enable_statistics",
        "coordinates_type",
    ],
    "universal": [
        "anisotropy_scaling",
        "anisotropy_angle",
        "drift_terms",
        "point_drift",
        "external_drift",
        "external_drift_x",
        "external_drift_y",
        "functional_drift",
    ],
    "ordinary3d": [
        "anisotropy_scaling_y",
        "anisotropy_scaling_z",
        "anisotropy_angle_x",
        "anisotropy_angle_y",
        "anisotropy_angle_z",
    ],
    "universal3d": [
        "anisotropy_scaling_y",
        "anisotropy_scaling_z",
        "anisotropy_angle_x",
        "anisotropy_angle_y",
        "anisotropy_angle_z",
        "drift_terms",
        "functional_drift",
    ],
}


def validate_method(method):
    """Validate the kriging method in use."""
    if method not in krige_methods.keys():
        raise ValueError(
            "Kriging method must be one of {}".format(krige_methods.keys())
        )


class Krige(RegressorMixin, BaseEstimator):
    """
    A scikit-learn wrapper class for Ordinary and Universal Kriging.

    This works with both Grid/RandomSearchCv for finding the best
    Krige parameters combination for a problem.

    Parameters
    ----------
    method: str, optional
        type of kriging to be performed
    variogram_model: str, optional
        variogram model to be used during Kriging
    nlags: int
        see OK/UK class description
    weight: bool
        see OK/UK class description
    n_closest_points: int
        number of closest points to be used during Ordinary Kriging
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
        method="ordinary",
        variogram_model="linear",
        nlags=6,
        weight=False,
        n_closest_points=10,
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
        validate_method(method)
        self.variogram_model = variogram_model
        self.variogram_parameters = variogram_parameters
        self.variogram_function = variogram_function
        self.nlags = nlags
        self.weight = weight
        self.verbose = verbose
        self.exact_values = exact_values
        self.pseudo_inv = pseudo_inv
        self.pseudo_inv_type = pseudo_inv_type
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        self.enable_statistics = enable_statistics
        self.coordinates_type = coordinates_type
        self.drift_terms = drift_terms
        self.point_drift = point_drift
        self.ext_drift_grid = ext_drift_grid
        self.functional_drift = functional_drift
        self.model = None  # not trained
        self.n_closest_points = n_closest_points
        self.method = method

    def fit(self, x, y, *args, **kwargs):
        """
        Fit the current model.

        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2) for 2d kriging
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (N, )
        """
        val_kw = "val" if self.method in threed_krige else "z"
        setup = dict(
            variogram_model=self.variogram_model,
            variogram_parameters=self.variogram_parameters,
            variogram_function=self.variogram_function,
            nlags=self.nlags,
            weight=self.weight,
            verbose=self.verbose,
            exact_values=self.exact_values,
            pseudo_inv=self.pseudo_inv,
            pseudo_inv_type=self.pseudo_inv_type,
        )
        add_setup = dict(
            anisotropy_scaling=self.anisotropy_scaling[0],
            anisotropy_angle=self.anisotropy_angle[0],
            enable_statistics=self.enable_statistics,
            coordinates_type=self.coordinates_type,
            anisotropy_scaling_y=self.anisotropy_scaling[0],
            anisotropy_scaling_z=self.anisotropy_scaling[1],
            anisotropy_angle_x=self.anisotropy_angle[0],
            anisotropy_angle_y=self.anisotropy_angle[1],
            anisotropy_angle_z=self.anisotropy_angle[2],
            drift_terms=self.drift_terms,
            point_drift=self.point_drift,
            external_drift=self.ext_drift_grid[0],
            external_drift_x=self.ext_drift_grid[1],
            external_drift_y=self.ext_drift_grid[2],
            functional_drift=self.functional_drift,
        )
        for kw in krige_methods_kws[self.method]:
            setup[kw] = add_setup[kw]
        input_kw = self._dimensionality_check(x)
        input_kw.update(setup)
        input_kw[val_kw] = y
        self.model = krige_methods[self.method](**input_kw)

    def _dimensionality_check(self, x, ext=""):
        if self.method in ("ordinary", "universal"):
            if x.shape[1] != 2:
                raise ValueError("2d krige can use only 2d points")
            else:
                return {"x" + ext: x[:, 0], "y" + ext: x[:, 1]}
        if self.method in ("ordinary3d", "universal3d"):
            if x.shape[1] != 3:
                raise ValueError("3d krige can use only 3d points")
            else:
                return {
                    "x" + ext: x[:, 0],
                    "y" + ext: x[:, 1],
                    "z" + ext: x[:, 2],
                }

    def predict(self, x, *args, **kwargs):
        """
        Predict.

        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2) for 2d kriging
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        Returns
        -------
        Prediction array
        """
        if not self.model:
            raise Exception("Not trained. Train first")
        points = self._dimensionality_check(x, ext="points")
        return self.execute(points, *args, **kwargs)[0]

    def execute(self, points, *args, **kwargs):
        # TODO array of Points, (x, y) pairs of shape (N, 2)
        """
        Execute.

        Parameters
        ----------
        points: dict

        Returns
        -------
        Prediction array
        Variance array
        """
        default_kw = dict(style="points", backend="loop")
        default_kw.update(kwargs)
        points.update(default_kw)
        if isinstance(self.model, (OrdinaryKriging, OrdinaryKriging3D)):
            points.update(dict(n_closest_points=self.n_closest_points))
        else:
            print("n_closest_points will be ignored for UniversalKriging")
        prediction, variance = self.model.execute(**points)
        return prediction, variance


def check_sklearn_model(model):
    """Check the sklearn method in use."""
    if not (isinstance(model, BaseEstimator) and isinstance(model, RegressorMixin)):
        raise RuntimeError(
            "Needs to supply an instance of a scikit-learn regression class."
        )


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
