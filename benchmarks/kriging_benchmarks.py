# -*- coding: utf-8 -*-
"""Benchmarks."""
from time import time
import numpy as np
from pykrige.ok import OrdinaryKriging
np.random.seed(19999)

VARIOGRAM_MODELS = ['power', 'gaussian', 'spherical',
                    'exponential', 'linear']
BACKENDS = ['vectorized', 'loop', 'C']
N_MOVING_WINDOW = [None, 10, 50, 100]


def make_benchark(n_train, n_test, n_dim=2):
    """Compute the benchmarks for Ordianry Kriging.

    Parameters
    ----------
    n_train : int
      number of points in the training set
    n_test : int
      number of points in the test set
    n_dim : int
      number of dimensions (default=2)

    Returns
    -------
    res : dict
      a dictionary with the timing results
    """
    X_train = np.random.rand(n_train, n_dim)
    y_train = np.random.rand(n_train)
    X_test = np.random.rand(n_test, n_dim)

    res = {}

    for variogram_model in VARIOGRAM_MODELS:
        tic = time()
        OK = OrdinaryKriging(X_train[:, 0], X_train[:, 1], y_train,
                             variogram_model='linear',
                             verbose=False, enable_plotting=False)
        res['t_train_{}'.format(variogram_model)] = time() - tic

    # All the following tests are performed with the linear variogram model
    for backend in BACKENDS:
        for n_closest_points in N_MOVING_WINDOW:

            if backend == 'vectorized' and n_closest_points is not None:
                continue  # this is not supported

            tic = time()
            OK.execute('points', X_test[:, 0], X_test[:, 1],
                       backend=backend,
                       n_closest_points=n_closest_points)
            res['t_test_{}_{}'.format(backend, n_closest_points)] = time() - tic

    return res


def print_benchmark(n_train, n_test, n_dim, res):
    """Print the benchmarks.

    Parameters
    ----------
    n_train : int
      number of points in the training set
    n_test : int
      number of points in the test set
    n_dim : int
      number of dimensions (default=2)
    res : dict
      a dictionary with the timing results
    """
    print('='*80)
    print(' '*10, 'N_dim={}, N_train={}, N_test={}'.format(n_dim,
                                                           n_train, n_test))
    print('='*80)
    print('\n', '# Training the model', '\n')
    print('|'.join(['{:>11} '.format(el) for el in ['t_train (s)'] +
                    VARIOGRAM_MODELS]))
    print('-' * (11 + 2) * (len(VARIOGRAM_MODELS) + 1))
    print('|'.join(['{:>11} '.format('Training')] +
                   ['{:>11.2} '.format(el) for el in
                    [res['t_train_{}'.format(mod)]
                     for mod in VARIOGRAM_MODELS]]))

    print('\n', '# Predicting kriging points', '\n')
    print('|'.join(['{:>11} '.format(el) for el in ['t_test (s)'] + BACKENDS]))
    print('-' * (11 + 2) * (len(BACKENDS) + 1))

    for n_closest_points in N_MOVING_WINDOW:
        timing_results = [res.get(
            't_test_{}_{}'.format(mod, n_closest_points), '')
                          for mod in BACKENDS]
        print('|'.join(['{:>11} '.format('N_nn=' + str(n_closest_points))] +
                       ['{:>11.2} '.format(el) for el in timing_results]))


if __name__ == '__main__':
    for no_train, no_test in [(400, 1000),
                              (400, 2000),
                              (800, 2000)]:
        results = make_benchark(no_train, no_test)
        print_benchmark(no_train, no_test, 2, results)
