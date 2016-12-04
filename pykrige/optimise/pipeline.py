import configparser
import csv
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pykrige.optimise.krige import Krige
config = configparser.ConfigParser()
config.read('pykrige/optimise/optimise.conf')


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
results = OrderedDict(estimator.cv_results_)

# the output file from optimisation conf
optimisation_output = config.get(section='output',
                                 option='optimisation_output')
out_keys = results.keys()

with open(optimisation_output, 'w') as csv_file:
    w = csv.DictWriter(csv_file, results.keys())
    w.writeheader()
    for r in zip(* results.values()):
        w.writerow(
            dict(zip(out_keys,
                     [str(x) if not isinstance(x, str) else x for x in r]))
        )

