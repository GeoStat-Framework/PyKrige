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

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging,
                 'ordinary3d': OrdinaryKriging3D,
                 'universal3d': UniversalKriging3D
                 }

threed_krige = ('ordinary3d', 'universal3d')


def validate_method(method):
    if method not in krige_methods.keys():
        raise ValueError('Kriging method must be '
                         'one of {}'.format(krige_methods.keys()))


class Krige(RegressorMixin, BaseEstimator):
    """
    A scikit-learn wrapper class for Ordinary and Universal Kriging.
    This works with both Grid/RandomSearchCv for finding the best
    Krige parameters combination for a problem.

    """

    def __init__(self,
                 method='ordinary',
                 variogram_model='linear',
                 nlags=6,
                 weight=False,
                 n_closest_points=10,
                 verbose=False):

        validate_method(method)
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.nlags = nlags
        self.weight = weight
        self.model = None  # not trained
        self.n_closest_points = n_closest_points
        self.method = method

    def fit(self, x, y, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2) for 2d kriging
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (N, )
        """

        points = self._dimensionality_check(x)

        # if condition required to address backward compatibility
        if self.method in threed_krige:
            self.model = krige_methods[self.method](
                val=y,
                variogram_model=self.variogram_model,
                nlags=self.nlags,
                weight=self.weight,
                verbose=self.verbose,
                **points
            )
        else:
            self.model = krige_methods[self.method](
                z=y,
                variogram_model=self.variogram_model,
                nlags=self.nlags,
                weight=self.weight,
                verbose=self.verbose,
                **points
            )

    def _dimensionality_check(self, x, ext=''):
        if self.method in ('ordinary', 'universal'):
            if x.shape[1] != 2:
                raise ValueError('2d krige can use only 2d points')
            else:
                return {'x' + ext: x[:, 0], 'y' + ext: x[:, 1]}
        if self.method in ('ordinary3d', 'universal3d'):
            if x.shape[1] != 3:
                raise ValueError('3d krige can use only 3d points')
            else:
                return {'x' + ext: x[:, 0],
                        'y' + ext: x[:, 1],
                        'z' + ext: x[:, 2]}

    def predict(self, x, *args, **kwargs):
        """
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
            raise Exception('Not trained. Train first')

        points = self._dimensionality_check(x, ext='points')

        return self.execute(points, *args, **kwargs)[0]

    def execute(self, points, *args, **kwargs):
        # TODO array of Points, (x, y) pairs of shape (N, 2)
        """
        Parameters
        ----------
        points: dict

        Returns:
        -------
        Prediction array
        Variance array
        """
        if isinstance(self.model, OrdinaryKriging) or \
                isinstance(self.model, OrdinaryKriging3D):
            prediction, variance = \
                self.model.execute('points',
                                   n_closest_points=self.n_closest_points,
                                   backend='loop',
                                   **points)
        else:
            print('n_closest_points will be ignored for UniversalKriging')
            prediction, variance = \
                self.model.execute('points', backend='loop', **points)

        return prediction, variance


def check_sklearn_model(model):
    if not (isinstance(model, BaseEstimator) and
            isinstance(model, RegressorMixin)):
        raise RuntimeError('Needs to supply an instance of a scikit-learn '
                           'regression class.')


class RegressionKriging:
    """
    This is an implementation of Regression-Kriging as described here:
    https://en.wikipedia.org/wiki/Regression-Kriging
    """

    def __init__(self,
                 regression_model=SVR(),
                 method='ordinary',
                 variogram_model='linear',
                 n_closest_points=10,
                 nlags=6,
                 weight=False,
                 verbose=False):
        """
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
        """
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
            )

    def fit(self, p, x, y):
        """
        fit the regression method and also Krige the residual

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
        print('Finished learning regression model')
        # residual=y-ml_pred
        self.krige.fit(x=x, y=y - ml_pred)
        print('Finished kriging residuals')

    def predict(self, p, x):
        """
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

        return self.krige_residual(x) + self.regression_model.predict(p)

    def krige_residual(self, x):
        """
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
        return self.krige.predict(x)

    def score(self, p, x, y, sample_weight=None):
        """
        Overloading default regression score method

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

        return r2_score(y_pred=self.predict(p, x),
                        y_true=y,
                        sample_weight=sample_weight)


