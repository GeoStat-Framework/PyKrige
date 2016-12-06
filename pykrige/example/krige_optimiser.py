import configparser
import numpy as np
from pykrige.compat import SKLEARN_INSTALLED, Pipeline, GridSearchCV, \
    validate_sklearn

if SKLEARN_INSTALLED:
    from pykrige.optimise import Krige

validate_sklearn()


config = configparser.ConfigParser()
config.read_string(
    """
    [hyperparameters]
    method = ordinary, universal
    variogram_model = linear, power, gaussian, spherical, exponential
    # insert other parameters
    """
    )


def setup_pipeline(config):
    steps = [('krige', Krige(verbose=True))]
    param_dict = {}
    if 'hyperparameters' in config:
        for k, v in config['hyperparameters'].items():
            param_dict['krige__' + k] = \
                [vv.strip() for vv in v.split(',')]

    pipe = Pipeline(steps=steps)
    estimator = GridSearchCV(pipe,
                             param_dict,
                             n_jobs=-1,
                             iid=False,
                             pre_dispatch='2*n_jobs',
                             verbose=True,
                             cv=5,
                             )
    return estimator

estimator = setup_pipeline(config)

# dummy data
X = np.random.randint(0, 400, size=(100, 2)).astype(float)
y = 5*np.random.rand(100)

# run the gridsearch
estimator.fit(X=X, y=y)
if hasattr(estimator, 'best_score_'):
    print(estimator.best_score_)
    print(estimator.best_params_)

if hasattr(estimator, 'cv_results_'):
    print(estimator.cv_results_['mean_test_score'])
    print(estimator.cv_results_['mean_train_score'])
    print(estimator.cv_results_['param_krige__method'])
    print(estimator.cv_results_['param_krige__variogram_model'])

