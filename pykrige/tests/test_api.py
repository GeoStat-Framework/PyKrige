from __future__ import absolute_import
from __future__ import print_function

from itertools import product

import numpy as np

from pykrige.rk import Krige
from pykrige.rk import threed_krige
from pykrige.compat import GridSearchCV

import pytest

from pykrige.compat import SKLEARN_INSTALLED
from decimal import Decimal

def _method_and_variogram():
    method = ['ordinary', 'universal', 'ordinary3d', 'universal3d']
    variogram_model = ['linear', 'power', 'gaussian', 'spherical',
                       'exponential']
    return product(method, variogram_model)


def test_krige():
    # dummy data
    np.random.seed(1)
    X = np.random.randint(0, 400, size=(20, 3)).astype(float)
    y = 5 * np.random.rand(20)

    for m, v in _method_and_variogram():
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


@pytest.mark.skipif(not SKLEARN_INSTALLED,
                    reason="scikit-learn not installed")
def test_gridsearch_cv_variogram_parameters():
    param_dict3d = {"method": ["ordinary3d"], "variogram_model": ["linear"],
                 "variogram_parameters": [{'slope': 1.0, 'nugget': 1.0},
                                          {'slope': 2.0, 'nugget': 1.0}]
               }

    estimator = GridSearchCV(Krige(), param_dict3d, verbose=True)

    # dummy data
    seed = np.random.RandomState(42)
    X3 = 400. * (1 + seed.rand(100, 3))
    y = 5 * seed.rand(100)

    # run the gridsearch
    estimator.fit(X=X3, y=y)

    # Expected best parameters and score
    best_params = [1.0,1.0]
    # best_score = -0.4624793735893478
    best_score = round(Decimal(-0.462479373589),12)

    if hasattr(estimator, 'best_params_'):
        assert len(estimator.cv_results_['param_variogram_parameters'])==len(param_dict3d["variogram_parameters"])
        for i,k in enumerate(estimator.best_params_["variogram_parameters"]):
            assert estimator.best_params_["variogram_parameters"][k] == best_params[i]
