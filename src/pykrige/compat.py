# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility."""
from pykrige.uk3d import UniversalKriging3D
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging

# sklearn
try:
    # keep train_test_split here for backward compatibility
    from sklearn.model_selection import train_test_split
    from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator

    SKLEARN_INSTALLED = True

except ImportError:
    SKLEARN_INSTALLED = False

    train_test_split = None

    class RegressorMixin:
        """Mock RegressorMixin."""

    class ClassifierMixin:
        """Mock ClassifierMixin."""

    class BaseEstimator:
        """Mock BaseEstimator."""


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


class SklearnException(Exception):
    """Exception for missing scikit-learn."""


def validate_method(method):
    """Validate the kriging method in use."""
    if method not in krige_methods.keys():
        raise ValueError(
            "Kriging method must be one of {}".format(krige_methods.keys())
        )


def validate_sklearn():
    """Validate presence of scikit-learn."""
    if not SKLEARN_INSTALLED:
        raise SklearnException(
            "sklearn needs to be installed in order to use this module"
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


def check_sklearn_model(model, task="regression"):
    """Check the sklearn method in use."""
    if task == "regression":
        if not (isinstance(model, BaseEstimator) and isinstance(model, RegressorMixin)):
            raise RuntimeError(
                "Needs to supply an instance of a scikit-learn regression class."
            )
    elif task == "classification":
        if not (
            isinstance(model, BaseEstimator) and isinstance(model, ClassifierMixin)
        ):
            raise RuntimeError(
                "Needs to supply an instance of a scikit-learn classification class."
            )
