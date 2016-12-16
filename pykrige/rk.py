import logging
from pykrige.compat import (RegressorMixin,
                            BaseEstimator,
                            r2_score,
                            SVR,
                            validate_sklearn
                            )

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

log = logging.getLogger(__name__)

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
                 verbose=False
                 ):

        validate_sklearn()
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
            log.warning('n_closest_points will be ignored for UniversalKriging')
            prediction, variance = \
                self.model.execute('points', x[:, 0], x[:, 1],
                                   backend='loop')

        return prediction, variance


def check_sklearn_model(model):
    if not (isinstance(model, BaseEstimator) and
            isinstance(model, RegressorMixin)):
        raise RuntimeError('Needs to supply an instance of a scikit-learn '
                           'regression class.')


class MLKrige(Krige):
    """
    This is an implementation of Regression-Kriging as described here:
    https://en.wikipedia.org/wiki/Regression-Kriging
    """

    def __init__(self,
                 ml_model=SVR(),
                 method='ordinary',
                 variogram_model='linear',
                 n_closest_points=10,
                 nlags=6,
                 weight=False,
                 verbose=False):
        """
        :param ml_model: machine learning model instance from sklearn
        :param method:
        :param variogram_model:
        :param n_closest_points:
        :param nlags:
        :param weight:
        :param verbose:
        """
        check_sklearn_model(ml_model)
        self.ml_model = ml_model
        self.n_closest_points = n_closest_points
        super(MLKrige, self).__init__(method=method,
                                      variogram_model=variogram_model,
                                      nlags=nlags,
                                      weight=weight,
                                      n_closest_points=n_closest_points,
                                      verbose=verbose,
                                      )

    def fit(self, x, lon_lat, y):
        """
        fit the ML method and also Krige the residual

        Parameters
        ----------
        x: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions)
            for ML regression
        lon_lat:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.
        y: ndarray
            array of targets (Nt, )
        """
        self.ml_model.fit(x, y)
        ml_pred = self.ml_model.predict(x)
        log.info('Finished learning regression model')
        # residual=y-ml_pred
        super(MLKrige, self).fit(x=lon_lat, y=y - ml_pred)
        log.info('Finished kriging residuals')

    def predict(self, x, lon_lat):
        """
        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions)
            for ML regression
        lon_lat:
            ndarray of (x, y) points. Needs to be a (Ns, 2) array
            corresponding to the lon/lat, for example.

        Returns
        -------
        pred: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).

        """

        return super(MLKrige, self).predict(lon_lat) + \
            self.ml_model.predict(x)

    def score(self, x, lon_lat, y, sample_weight=None):
        """
        Overloading default regression score method
        """

        return r2_score(y_pred=self.predict(x, lon_lat),
                        y_true=y,
                        sample_weight=sample_weight)


