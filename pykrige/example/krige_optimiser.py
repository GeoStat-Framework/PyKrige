import configparser
import csv
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pykrige.optimise import Krige

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
print(OrderedDict(estimator.cv_results_))
