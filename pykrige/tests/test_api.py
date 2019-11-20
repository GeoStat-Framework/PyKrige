from __future__ import absolute_import
from __future__ import print_function

from itertools import product

import numpy as np

from pykrige.rk import Krige
from pykrige.rk import threed_krige
from pykrige.compat import GridSearchCV


def _method_and_vergiogram():
    method = ['ordinary', 'universal', 'ordinary3d', 'universal3d']
    variogram_model = ['linear', 'power', 'gaussian', 'spherical',
                       'exponential']
    return product(method, variogram_model)


def test_krige():
    # dummy data
    np.random.seed(1)
    X = np.random.randint(0, 400, size=(20, 3)).astype(float)
    y = 5 * np.random.rand(20)

    for m, v in _method_and_vergiogram():
        param_dict = {'method': [m], 'variogram_model': [v]}

        estimator = GridSearchCV(Krige(),
                                 param_dict,
                                 n_jobs=-1,
                                 iid=False,
                                 pre_dispatch='2*n_jobs',
                                 verbose=False,
                                 cv=5,
                                 )
        # run the gridsearch
        if m in ['ordinary', 'universal']:
            estimator.fit(X=X[:, :2], y=y)
        else:
            estimator.fit(X=X, y=y)
        if hasattr(estimator, 'best_score_'):
            if m in threed_krige:
                assert estimator.best_score_ > -10.0
            else:
                assert estimator.best_score_ > -3.0
        if hasattr(estimator, 'cv_results_'):
            assert estimator.cv_results_['mean_train_score'] > 0
