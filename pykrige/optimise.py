import numpy as np
import logging
from scipy.stats import norm
from pykrige.compat import (RegressorMixin,
                            BaseEstimator,
                            validate_sklearn
                            )

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

log = logging.getLogger(__name__)

krige_methods = {'ordinary': OrdinaryKriging,
                 'universal': UniversalKriging}


class ConfigException(Exception):
    pass


def validate_method(method):
    if method not in krige_methods.keys():
        raise ConfigException('Kirging method must be '
                              'one of {}'.format(krige_methods.keys()))


class TagsMixin():
    """
    Mixin class to aid a pipeline in establishing the types of predictive
    outputs to be expected from the ML algorithms in this module.
    """

    def get_predict_tags(self):
        """
        Get the types of prediction outputs from this algorithm.

        Returns
        -------
        list:
            of strings with the types of outputs that can be returned by this
            algorithm. This depends on the prediction methods implemented (e.g.
            ``predict``, ``predict_proba``).
        """

        tags = ['Prediction']
        if hasattr(self, 'predict_proba'):
            tags.extend(['Variance', 'Lower quantile', 'Upper quantile'])

        return tags


class KrigePredictProbaMixin():
    """
    Mixin class for providing a ``predict_proba`` method to the
    Krige class.

    This is especially for use with PyKrige Ordinary/UniversalKriging classes.
    """
    def predict_proba(self, x, interval=0.95, *args, **kwargs):
        """
        Predictive mean and variance for a probabilistic regressor.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query dataset (Ns samples, d dimensions).
        interval: float, optional
            The percentile confidence interval (e.g. 95%) to return.
        Returns
        -------
        Ey: ndarray
            The expected value of ys for the query inputs, X of shape (Ns,).
        Vy: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, X of shape (Ns,).
        ql: ndarray
            The lower end point of the interval with shape (Ns,)
        qu: ndarray
            The upper end point of the interval with shape (Ns,)
        """
        prediction, variance = \
            self.model.execute('points', x[:, 0], x[:, 1])

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=prediction,
                               scale=np.sqrt(variance))

        return prediction, variance, ql, qu


class Krige(TagsMixin, RegressorMixin, BaseEstimator, KrigePredictProbaMixin):
    """
    A scikit-learn wrapper class for Ordinary and Universal Kriging.
    This works for both Grid/RandomSearchCv for optimising the
    Krige parameters.

    """

    def __init__(self,
                 method='ordinary',
                 variogram_model='linear',
                 verbose=False
                 ):

        validate_sklearn()
        validate_method(method)
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.model = None  # not trained
        self.method = method

    def fit(self, x, y, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs
        y: ndarray
            array of targets
        """
        if x.shape[1] != 2:
            raise ConfigException('krige can use only 2 covariates')

        self.model = krige_methods[self.method](
            x=x[:, 0],
            y=x[:, 1],
            z=y,
            variogram_model=self.variogram_model,
            verbose=self.verbose
         )

    def predict(self, x, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray

        Returns:
        -------
        Prediction array
        """
        if not self.model:
            raise Exception('Not trained. Train first')

        return self.model.execute('points', x[:, 0], x[:, 1])[0]