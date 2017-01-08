# coding: utf-8
from pykrige.compat import validate_sklearn
validate_sklearn()
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.svm import SVR
from sklearn.metrics import r2_score

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}


def validate_method(method):
    if method not in krige_methods.keys():
        raise ValueError('Kirging method must be '
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
            array of Points, (x, y) pairs, (x, y) pairs of shape (Nt, 2)
        y: ndarray
            array of targets (Nt, )
        """
        if x.shape[1] != 2:
            raise ValueError('krige can use only 2 covariates')

        self.model = krige_methods[self.method](
            x=x[:, 0],
            y=x[:, 1],
            z=y,
            variogram_model=self.variogram_model,
            nlags=self.nlags,
            weight=self.weight,
            verbose=self.verbose
         )

    def predict(self, x, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2)

        Returns:
        -------
        Prediction array
        """
        if not self.model:
            raise Exception('Not trained. Train first')

        return self.execute(x, *args, **kwargs)[0]

    def execute(self, x, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2)

        Returns:
        -------
        Prediction array
        Variance array
        """
        if isinstance(self.model, OrdinaryKriging):
            prediction, variance = \
                self.model.execute('points', x[:, 0], x[:, 1],
                                   n_closest_points=self.n_closest_points,
                                   backend='loop')
        else:
            print('n_closest_points will be ignored for UniversalKriging')
            prediction, variance = \
                self.model.execute('points', x[:, 0], x[:, 1],
                                   backend='loop')

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
        :param regression_model: machine learning model instance from sklearn
        :param method:
        :param variogram_model:
        :param n_closest_points:
        :param nlags:
        :param weight:
        :param verbose:
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
            (Ns, d) array of predictor variables (Nt samples, d dimensions)
            for regression
        x:
            ndarray of (x, y) points. Needs to be a (Nt, 2) array
            corresponding to the lon/lat, for example.
        y: ndarray
            array of targets (Nt, )
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
        x:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.

        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, of shape (Ns,).

        """

        return self.krige_residual(x) + \
               self.regression_model.predict(p)

    def krige_residual(self, x):
        """
        :param x:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
        :return:
        residual: ndarray
            kriged residual values
        """
        return self.krige.predict(x)

    def score(self, p, x, y, sample_weight=None):
        """
        Overloading default regression score method
        """

        return r2_score(y_pred=self.predict(p, x),
                        y_true=y,
                        sample_weight=sample_weight)


