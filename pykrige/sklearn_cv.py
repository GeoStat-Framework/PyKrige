import logging
from pykrige.compat import (RegressorMixin,
                            BaseEstimator,
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
    This works for both Grid/RandomSearchCv for optimizing the
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
            raise ValueError('krige can use only 2 covariates')

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
