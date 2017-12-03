from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""
Testing code.
Updated BSM February 2017
"""

import os
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pykrige import kriging_tools as kt
from pykrige import core
from pykrige import variogram_models
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Result(unittest.TestCase):
    pass


allclose_pars = {'rtol': 1e-05, 'atol': 1e-08}


@pytest.fixture
def self():

    self = Result()

    self.test_data = np.genfromtxt(os.path.join(BASE_DIR,
                                                'test_data/test_data.txt'))
    self.ok_test_answer, self.ok_test_gridx, self.ok_test_gridy, \
        cellsize, no_data = kt.read_asc_grid(
            os.path.join(BASE_DIR, 'test_data/test1_answer.asc'),
                footer=2)
    self.uk_test_answer, self.uk_test_gridx, self.uk_test_gridy, \
        cellsize, no_data = kt.read_asc_grid(
            os.path.join(BASE_DIR, 'test_data/test2_answer.asc'),
                footer=2)

    self.simple_data = np.array([[0.3, 1.2, 0.47],
                                 [1.9, 0.6, 0.56],
                                 [1.1, 3.2, 0.74],
                                 [3.3, 4.4, 1.47],
                                 [4.7, 3.8, 1.74]])
    self.simple_gridx = np.arange(0.0, 6.0, 1.0)
    self.simple_gridx_2 = np.arange(0.0, 5.5, 0.5)
    self.simple_gridy = np.arange(0.0, 5.5, 0.5)
    xi, yi = np.meshgrid(self.simple_gridx, self.simple_gridy)
    self.mask = np.array(xi == yi)

    self.simple_data_3d = np.array([[0.1, 0.1, 0.3, 0.9],
                                    [0.2, 0.1, 0.4, 0.8],
                                    [0.1, 0.3, 0.1, 0.9],
                                    [0.5, 0.4, 0.4, 0.5],
                                    [0.3, 0.3, 0.2, 0.7]])
    self.simple_gridx_3d = np.arange(0.0, 0.6, 0.05)
    self.simple_gridy_3d = np.arange(0.0, 0.6, 0.01)
    self.simple_gridz_3d = np.arange(0.0, 0.6, 0.1)
    zi, yi, xi = np.meshgrid(self.simple_gridz_3d, self.simple_gridy_3d,
                             self.simple_gridx_3d, indexing='ij')
    self.mask_3d = np.array((xi == yi) & (yi == zi))
    return self


def test_core_adjust_for_anisotropy(self):

    X = np.array([[1.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0, 0.0, -1.0]]).T
    X_adj = core._adjust_for_anisotropy(X, [0.0, 0.0], [2.0], [90.0])
    assert_allclose(X_adj[:, 0], np.array([0.0, 1.0, 0.0, -1.0]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([-2.0, 0.0, 2.0, 0.0]), **allclose_pars)


def test_core_adjust_for_anisotropy_3d(self):

    # this is a bad examples, as the X matrix is symmetric
    # and insensitive to transpositions
    X = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]).T
    X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [90., 0., 0.])
    assert_allclose(X_adj[:, 0], np.array([1., 0., 0.]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([0., 0., 2.]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([0., -2., 0.]), **allclose_pars)
    X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [0., 90., 0.])
    assert_allclose(X_adj[:, 0], np.array([0., 0., -1.]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([0., 2., 0.]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([2., 0., 0.]), **allclose_pars)
    X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [0., 0., 90.])
    assert_allclose(X_adj[:, 0], np.array([0., 1., 0.]), **allclose_pars)
    assert_allclose(X_adj[:, 1], np.array([-2., 0., 0.]), **allclose_pars)
    assert_allclose(X_adj[:, 2], np.array([0., 0., 2.]), **allclose_pars)


def test_core_make_variogram_parameter_list(self):

    # test of first case - variogram_model_parameters is None
    # function should return None unaffected
    result = core._make_variogram_parameter_list('linear', None)
    assert result is None

    # tests for second case - variogram_model_parameters is dict
    with pytest.raises(KeyError):
        core._make_variogram_parameter_list(
                      'linear', {'tacos': 1., 'burritos': 2.})
    result = core._make_variogram_parameter_list('linear', {'slope': 1.,
                                                            'nugget': 0.})
    assert result == [1., 0.]

    with pytest.raises(KeyError):
        core._make_variogram_parameter_list(
                      'power', {'frijoles': 1.})
    result = core._make_variogram_parameter_list('power', {'scale': 2.,
                                                           'exponent': 1.,
                                                           'nugget': 0.})
    assert result == [2., 1., 0.]

    with pytest.raises(KeyError):
        core._make_variogram_parameter_list(
                      'exponential', {'tacos': 1.})
    with pytest.raises(KeyError):
            core._make_variogram_parameter_list(
                      'exponential', {'range': 1., 'nugget': 1.})
    result = core._make_variogram_parameter_list('exponential',
                                                 {'sill': 5., 'range': 2.,
                                                  'nugget': 1.})
    assert result == [4., 2., 1.]
    result = core._make_variogram_parameter_list('exponential',
                                                 {'psill': 4., 'range': 2.,
                                                  'nugget': 1.})
    assert result == [4., 2., 1.]

    with pytest.raises(TypeError):
        core._make_variogram_parameter_list('custom', {'junk': 1.})
    with pytest.raises(ValueError):
        core._make_variogram_parameter_list('blarg', {'junk': 1.})

    # tests for third case - variogram_model_parameters is list
    with pytest.raises(ValueError):
        core._make_variogram_parameter_list(
                      'linear', [1., 2., 3.])
    result = core._make_variogram_parameter_list('linear', [1., 2.])
    assert result == [1., 2.]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list('power', [1., 2.])

    result = core._make_variogram_parameter_list('power', [1., 2., 3.])
    assert result == [1., 2., 3.]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list(
                      'exponential', [1., 2., 3., 4.])
    result = core._make_variogram_parameter_list('exponential',
                                                 [5., 2., 1.])
    assert result == [4., 2., 1.]

    result = core._make_variogram_parameter_list('custom', [1., 2., 3.])
    assert result == [1., 2., 3]

    with pytest.raises(ValueError):
        core._make_variogram_parameter_list('junk', [1., 1., 1.])

    # test for last case - make sure function handles incorrect
    # variogram_model_parameters type appropriately
    with pytest.raises(TypeError):
        core._make_variogram_parameter_list('linear', 'tacos')


def test_core_initialize_variogram_model(self):

    # Note the variogram_function argument is not a string in real life...
    # core._initialize_variogram_model also checks the length of input
    # lists, which is redundant now because the same tests are done in
    # core._make_variogram_parameter_list
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
                      np.vstack((self.test_data[:, 0],
                                 self.test_data[:, 1])).T,
                      self.test_data[:, 2], 'linear', [0.0], 'linear',
                      6, False, 'euclidean')
    with pytest.raises(ValueError):
            core._initialize_variogram_model(
                      np.vstack((self.test_data[:, 0],
                                 self.test_data[:, 1])).T,
                      self.test_data[:, 2], 'spherical', [0.0],
                      'spherical', 6, False, 'euclidean')

    # core._initialize_variogram_model does also check coordinate type,
    # this is NOT redundant
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
                      np.vstack((self.test_data[:, 0],
                                 self.test_data[:, 1])).T,
                      self.test_data[:, 2], 'spherical', [0.0, 0.0, 0.0],
                      'spherical', 6, False, 'tacos')

    x = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
    y = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
    z = np.arange(1.0, 5.0, 1.0)
    lags, semivariance, variogram_model_parameters = \
        core._initialize_variogram_model(np.vstack((x, y)).T, z,
                                         'linear', [0.0, 0.0],
                                         'linear', 6, False, 'euclidean')

    assert_allclose(lags, np.array([1.0, 2.0, 3.0]))
    assert_allclose(semivariance, np.array([0.5, 2.0, 4.5]))


def test_core_initialize_variogram_model_3d(self):

    # Note the variogram_function argument is not a string in real life...
    # again, these checks in core._initialize_variogram_model are redundant
    # now because the same tests are done in
    # core._make_variogram_parameter_list
    with pytest.raises(ValueError):
        core._initialize_variogram_model(
                      np.vstack((self.simple_data_3d[:, 0],
                                 self.simple_data_3d[:, 1],
                                 self.simple_data_3d[:, 2])).T,
                      self.simple_data_3d[:, 3], 'linear', [0.0], 'linear',
                      6, False, 'euclidean')

    with pytest.raises(ValueError):
            core._initialize_variogram_model(
                      np.vstack((self.simple_data_3d[:, 0],
                                 self.simple_data_3d[:, 1],
                                 self.simple_data_3d[:, 2])).T,
                      self.simple_data_3d[:, 3], 'spherical', [0.0],
                      'spherical', 6, False, 'euclidean')

    with pytest.raises(ValueError):
                core._initialize_variogram_model(
                      np.vstack((self.simple_data_3d[:, 0],
                                 self.simple_data_3d[:, 1],
                                 self.simple_data_3d[:, 2])).T,
                      self.simple_data_3d[:, 3], 'linear', [0.0, 0.0],
                      'linear', 6, False, 'geographic')

    lags, semivariance, variogram_model_parameters = \
        core._initialize_variogram_model(
            np.vstack((np.array([1., 2., 3., 4.]),
                       np.array([1., 2., 3., 4.]),
                       np.array([1., 2., 3., 4.]))).T,
            np.array([1., 2., 3., 4.]), 'linear', [0.0, 0.0], 'linear', 3,
            False, 'euclidean')
    assert_allclose(lags, np.array([np.sqrt(3.), 2.*np.sqrt(3.),
                                                3.*np.sqrt(3.)]))
    assert_allclose(semivariance, np.array([0.5, 2.0, 4.5]))


def test_core_calculate_variogram_model(self):

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.05, 2.95, 4.05, 4.95]),
        'linear', variogram_models.linear_variogram_model, False)
    assert_allclose(res, np.array([0.98, 1.05]), 0.01, 0.01)

    res = core._calculate_variogram_model(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.05, 2.95, 4.05, 4.95]),
        'linear', variogram_models.linear_variogram_model, True)
    assert_allclose(res, np.array([0.98, 1.05]), 0.01, 0.01)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([1., 2.8284271,
                                              5.1961524, 8.]),
        'power', variogram_models.power_variogram_model, False)
    assert_allclose(res, np.array([1., 1.5, 0.]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([1.0, 1.4142, 1.7321, 2.0]),
        'power', variogram_models.power_variogram_model, False)
    assert_allclose(res, np.array([1., 0.5, 0.]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([1.2642, 1.7293,
                                              1.9004, 1.9634]),
        'exponential', variogram_models.exponential_variogram_model, False)
    assert_allclose(res, np.array([2., 3., 0.]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([0.5769, 1.4872,
                                              1.9065, 1.9914]),
        'gaussian', variogram_models.gaussian_variogram_model, False)
    assert_allclose(res, np.array([2., 3., 0.]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([3.33060952, 3.85063879,
                                              3.96667301, 3.99256374]),
        'exponential', variogram_models.exponential_variogram_model, False)
    assert_allclose(res, np.array([3., 2., 1.]), 0.001, 0.001)

    res = core._calculate_variogram_model(
        np.array([1., 2., 3., 4.]), np.array([2.60487044, 3.85968813,
                                              3.99694817, 3.99998564]),
        'gaussian', variogram_models.gaussian_variogram_model, False)
    assert_allclose(res, np.array([3., 2., 1.]), 0.001, 0.001)


def test_core_krige(self):

    # Example 3.2 from Kitanidis
    data = np.array([[9.7, 47.6, 1.22],
                     [43.8, 24.6, 2.822]])
    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1])).T, data[:, 2],
                        np.array([18.8, 67.9]),
                        variogram_models.linear_variogram_model,
                        [0.006, 0.1], 'euclidean')
    self.assertAlmostEqual(z, 1.6364, 4)
    self.assertAlmostEqual(ss, 0.4201, 4)

    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1])).T, data[:, 2],
                        np.array([43.8, 24.6]),
                        variogram_models.linear_variogram_model,
                        [0.006, 0.1], 'euclidean')
    self.assertAlmostEqual(z, 2.822, 3)
    self.assertAlmostEqual(ss, 0.0, 3)


def test_core_krige_3d(self):

    # Adapted from example 3.2 from Kitanidis
    data = np.array([[9.7, 47.6, 1.0, 1.22],
                     [43.8, 24.6, 1.0, 2.822]])
    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
                        data[:, 3], np.array([18.8, 67.9, 1.0]),
                        variogram_models.linear_variogram_model,
                        [0.006, 0.1], 'euclidean')
    self.assertAlmostEqual(z, 1.6364, 4)
    self.assertAlmostEqual(ss, 0.4201, 4)

    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1], data[:, 2])).T,
                        data[:, 3], np.array([43.8, 24.6, 1.0]),
                        variogram_models.linear_variogram_model,
                        [0.006, 0.1], 'euclidean')
    self.assertAlmostEqual(z, 2.822, 3)
    self.assertAlmostEqual(ss, 0.0, 3)


def test_ok(self):

    # Test to compare OK results to those obtained using KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

    ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                         variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0])
    z, ss = ok.execute('grid', self.ok_test_gridx, self.ok_test_gridy, backend='vectorized')
    assert_allclose(z, self.ok_test_answer)
    z, ss = ok.execute('grid', self.ok_test_gridx, self.ok_test_gridy, backend='loop')
    assert_allclose(z, self.ok_test_answer)


def test_ok_update_variogram_model(self):

    with pytest.raises(ValueError):
        OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1],
                      self.test_data[:, 2], variogram_model='blurg')

    ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2])
    variogram_model = ok.variogram_model
    variogram_parameters = ok.variogram_model_parameters
    anisotropy_scaling = ok.anisotropy_scaling
    anisotropy_angle = ok.anisotropy_angle

    with pytest.raises(ValueError):
        ok.update_variogram_model('blurg')

    ok.update_variogram_model('power', anisotropy_scaling=3.0, anisotropy_angle=45.0)
    assert variogram_model != ok.variogram_model
    assert variogram_parameters != ok.variogram_model_parameters
    assert anisotropy_scaling != ok.anisotropy_scaling
    assert anisotropy_angle != ok.anisotropy_angle


def test_ok_execute(self):

    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])

    with pytest.raises(ValueError):
        ok.execute('blurg', self.simple_gridx, self.simple_gridy)

    z, ss = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='vectorized')
    shape = (self.simple_gridy.size, self.simple_gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z, ss = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
    shape = (self.simple_gridy.size, self.simple_gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    with pytest.raises(IOError):
        ok.execute('masked', self.simple_gridx, self.simple_gridy, backend='vectorized')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=mask,
                   backend='vectorized')
    z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(IOError):
        ok.execute('masked', self.simple_gridx, self.simple_gridy, backend='loop')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=mask,
                   backend='loop')
    z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        ok.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                   backend='vectorized')
    z, ss = ok.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='vectorized')
    assert z.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        ok.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                   backend='loop')
    z, ss = ok.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='loop')
    assert z.shape == (1,)
    assert ss.shape == (1,)


def test_cython_ok(self):
    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
    z1, ss1 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
    z2, ss2 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='C')
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)

    closest_points = 4

    z1, ss1 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop',
                         n_closest_points=closest_points)
    z2, ss2 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='C',
                         n_closest_points=closest_points)
    assert_allclose(z1, z2)
    assert_allclose(ss1, ss2)


def test_uk(self):

    # Test to compare UK with linear drift to results from KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

    uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                          variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0],
                          drift_terms=['regional_linear'])
    z, ss = uk.execute('grid', self.uk_test_gridx, self.uk_test_gridy, backend='vectorized')
    assert_allclose(z, self.uk_test_answer)
    z, ss = uk.execute('grid', self.uk_test_gridx, self.uk_test_gridy, backend='loop')
    assert_allclose(z, self.uk_test_answer)


def test_uk_update_variogram_model(self):

    with pytest.raises(ValueError):
        UniversalKriging(self.test_data[:, 0], self.test_data[:, 1],
                      self.test_data[:, 2], variogram_model='blurg')
    with pytest.raises(ValueError):
        UniversalKriging(self.test_data[:, 0], self.test_data[:, 1],
                      self.test_data[:, 2], drift_terms=['external_Z'])
    with pytest.raises(ValueError):
        UniversalKriging(self.test_data[:, 0], self.test_data[:, 1],
                      self.test_data[:, 2], drift_terms=['external_Z'], external_drift=np.array([0]))
    with pytest.raises(ValueError):
        UniversalKriging(self.test_data[:, 0], self.test_data[:, 1],
                      self.test_data[:, 2], drift_terms=['point_log'])

    uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2])
    variogram_model = uk.variogram_model
    variogram_parameters = uk.variogram_model_parameters
    anisotropy_scaling = uk.anisotropy_scaling
    anisotropy_angle = uk.anisotropy_angle

    with pytest.raises(ValueError):
        uk.update_variogram_model('blurg')
    uk.update_variogram_model('power', anisotropy_scaling=3.0, anisotropy_angle=45.0)
    assert variogram_model != uk.variogram_model
    assert variogram_parameters != uk.variogram_model_parameters
    assert anisotropy_scaling != uk.anisotropy_scaling
    assert anisotropy_angle != uk.anisotropy_angle


def test_uk_calculate_data_point_zscalars(self):

    dem = np.arange(0.0, 5.1, 0.1)
    dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
    dem_x = np.arange(0.0, 5.1, 0.1)
    dem_y = np.arange(0.0, 6.0, 1.0)

    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                      drift_terms=['external_Z'])
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                      drift_terms=['external_Z'], external_drift=dem)
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                      drift_terms=['external_Z'], external_drift=dem, external_drift_x=dem_x,
                      external_drift_y=np.arange(0.0, 5.0, 1.0))

    uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                          variogram_model='linear', variogram_parameters=[1.0, 0.0],
                          drift_terms=['external_Z'], external_drift=dem, external_drift_x=dem_x,
                          external_drift_y=dem_y)
    assert_allclose(uk.z_scalars, self.simple_data[:, 0])

    xi, yi = np.meshgrid(np.arange(0.0, 5.3, 0.1), self.simple_gridy)
    with pytest.raises(ValueError):
        uk._calculate_data_point_zscalars(xi, yi)

    xi, yi = np.meshgrid(np.arange(0.0, 5.0, 0.1), self.simple_gridy)
    z_scalars = uk._calculate_data_point_zscalars(xi, yi)
    assert_allclose(z_scalars[0, :], np.arange(0.0, 5.0, 0.1))


def test_uk_execute_single_point(self):

    # Test data and answer from lecture notes by Nicolas Christou, UCLA Stats
    data = np.array([[61.0, 139.0, 477.0],
                     [63.0, 140.0, 696.0],
                     [64.0, 129.0, 227.0],
                     [68.0, 128.0, 646.0],
                     [71.0, 140.0, 606.0],
                     [73.0, 141.0, 791.0],
                     [75.0, 128.0, 783.0]])
    point = (65.0, 137.0)
    z_answer = 567.54
    ss_answer = 9.044

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='exponential',
                          variogram_parameters=[10.0, 9.99, 0.0], drift_terms=['regional_linear'])
    z, ss = uk.execute('points', np.array([point[0]]), np.array([point[1]]), backend='vectorized')
    self.assertAlmostEqual(z_answer, z[0], places=0)
    self.assertAlmostEqual(ss_answer, ss[0], places=0)

    z, ss = uk.execute('points', np.array([61.0]), np.array([139.0]), backend='vectorized')
    self.assertAlmostEqual(z[0], 477.0, 3)
    self.assertAlmostEqual(ss[0], 0.0, 3)

    z, ss = uk.execute('points', np.array([61.0]), np.array([139.0]), backend='loop')
    self.assertAlmostEqual(z[0], 477.0, 3)
    self.assertAlmostEqual(ss[0], 0.0, 3)


def test_uk_execute(self):

    uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                          variogram_model='linear', drift_terms=['regional_linear'])

    with pytest.raises(ValueError):
        uk.execute('blurg', self.simple_gridx, self.simple_gridy)
    with pytest.raises(ValueError):
        uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='mrow')

    z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='vectorized')
    shape = (self.simple_gridy.size, self.simple_gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
    shape = (self.simple_gridy.size, self.simple_gridx.size)
    assert z.shape == shape
    assert ss.shape == shape
    assert np.amax(z) != np.amin(z)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(z)

    with pytest.raises(IOError):
        uk.execute('masked', self.simple_gridx, self.simple_gridy, backend='vectorized')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=mask,
                   backend='vectorized')
    z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(IOError):
        uk.execute('masked', self.simple_gridx, self.simple_gridy, backend='loop')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=mask,
                   backend='loop')
    z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked
    z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0] is np.ma.masked
    assert ss[0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        uk.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                   backend='vectorized')
    z, ss = uk.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='vectorized')
    assert z.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        uk.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                   backend='loop')
    z, ss = uk.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='loop')
    assert z.shape == (1,)
    assert ss.shape == (1,)


def test_ok_uk_produce_same_result(self):

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                         variogram_model='linear', verbose=False, enable_plotting=False)
    z_ok, ss_ok = ok.execute('grid', gridx, gridy, backend='vectorized')
    uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                          variogram_model='linear', verbose=False, enable_plotting=False)
    z_uk, ss_uk = uk.execute('grid', gridx, gridy, backend='vectorized')
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)

    z_ok, ss_ok = ok.execute('grid', gridx, gridy, backend='loop')
    z_uk, ss_uk = uk.execute('grid', gridx, gridy, backend='loop')
    assert_allclose(z_ok, z_uk)
    assert_allclose(ss_ok, ss_uk)


def test_ok_backends_produce_same_result(self):

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                         variogram_model='linear', verbose=False, enable_plotting=False)
    z_ok_v, ss_ok_v = ok.execute('grid', gridx, gridy, backend='vectorized')
    z_ok_l, ss_ok_l = ok.execute('grid', gridx, gridy, backend='loop')
    assert_allclose(z_ok_v, z_ok_l)
    assert_allclose(ss_ok_v, ss_ok_l)


def test_uk_backends_produce_same_result(self):

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                          variogram_model='linear', verbose=False, enable_plotting=False)
    z_uk_v, ss_uk_v = uk.execute('grid', gridx, gridy, backend='vectorized')
    z_uk_l, ss_uk_l = uk.execute('grid', gridx, gridy, backend='loop')
    assert_allclose(z_uk_v, z_uk_l)
    assert_allclose(ss_uk_v, ss_uk_l)


def test_kriging_tools(self):

    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
    z_write, ss_write = ok.execute('grid', self.simple_gridx, self.simple_gridy)

    kt.write_asc_grid(self.simple_gridx, self.simple_gridy, z_write,
                      filename=os.path.join(BASE_DIR, 'test_data/temp.asc'), style=1)
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(BASE_DIR, 'test_data/temp.asc'))
    assert_allclose(z_write, z_read, 0.01, 0.01)
    assert_allclose(self.simple_gridx, x_read)
    assert_allclose(self.simple_gridy, y_read)

    z_write, ss_write = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask)
    kt.write_asc_grid(self.simple_gridx, self.simple_gridy, z_write,
                      filename=os.path.join(BASE_DIR, 'test_data/temp.asc'), style=1)
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(BASE_DIR, 'test_data/temp.asc'))
    assert (np.ma.allclose(z_write, np.ma.masked_where(z_read == no_data, z_read),
                                   masked_equal=True, rtol=0.01, atol=0.01))
    assert_allclose(self.simple_gridx, x_read)
    assert_allclose(self.simple_gridy, y_read)

    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
    z_write, ss_write = ok.execute('grid', self.simple_gridx_2, self.simple_gridy)

    kt.write_asc_grid(self.simple_gridx_2, self.simple_gridy, z_write,
                      filename=os.path.join(BASE_DIR, 'test_data/temp.asc'), style=2)
    z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(BASE_DIR, 'test_data/temp.asc'))
    assert_allclose(z_write, z_read, 0.01, 0.01)
    assert_allclose(self.simple_gridx_2, x_read)
    assert_allclose(self.simple_gridy, y_read)

    os.remove(os.path.join(BASE_DIR, 'test_data/temp.asc'))


def test_uk_three_primary_drifts(self):

    well = np.array([[1.1, 1.1, -1.0]])
    dem = np.arange(0.0, 5.1, 0.1)
    dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
    dem_x = np.arange(0.0, 5.1, 0.1)
    dem_y = np.arange(0.0, 6.0, 1.0)

    uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                          variogram_model='linear', drift_terms=['regional_linear', 'external_Z', 'point_log'],
                          point_drift=well, external_drift=dem, external_drift_x=dem_x, external_drift_y=dem_y)

    z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='vectorized')
    assert z.shape == (self.simple_gridy.shape[0], self.simple_gridx.shape[0])
    assert ss.shape == (self.simple_gridy.shape[0], self.simple_gridx.shape[0])
    assert np.all(np.isfinite(z))
    assert not np.all(np.isnan(z))
    assert np.all(np.isfinite(ss))
    assert not np.all(np.isnan(ss))

    z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
    assert z.shape == (self.simple_gridy.shape[0], self.simple_gridx.shape[0])
    assert ss.shape == (self.simple_gridy.shape[0], self.simple_gridx.shape[0])
    assert np.all(np.isfinite(z))
    assert not np.all(np.isnan(z))
    assert np.all(np.isfinite(ss))
    assert not np.all(np.isnan(ss))


def test_uk_specified_drift(self):

    xg, yg = np.meshgrid(self.simple_gridx, self.simple_gridy)
    well = np.array([[1.1, 1.1, -1.0]])
    point_log = well[0, 2] * np.log(np.sqrt((xg - well[0, 0])**2. + (yg - well[0, 1])**2.)) * -1.
    if np.any(np.isinf(point_log)):
        point_log[np.isinf(point_log)] = -100. * well[0, 2] * -1.
    point_log_data = well[0, 2] * np.log(np.sqrt((self.simple_data[:, 0] - well[0, 0])**2. +
                                                 (self.simple_data[:, 1] - well[0, 1])**2.)) * -1.
    if np.any(np.isinf(point_log_data)):
        point_log_data[np.isinf(point_log_data)] = -100. * well[0, 2] * -1.

    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'])
    with pytest.raises(TypeError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'],
                      specified_drift=self.simple_data[:, 0])
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                      self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'],
                      specified_drift=[self.simple_data[:2, 0]])

    uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['specified'],
                               specified_drift=[self.simple_data[:, 0], self.simple_data[:, 1]])
    with pytest.raises(ValueError):
        uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                        specified_drift_arrays=[self.simple_gridx, self.simple_gridy])
    with pytest.raises(TypeError):
        uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                        specified_drift_arrays=self.simple_gridx)
    with pytest.raises(ValueError):
        uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                        specified_drift_arrays=[xg])
    z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy, specified_drift_arrays=[xg, yg])

    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear'])
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)

    uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['specified'],
                               specified_drift=[point_log_data])
    z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                                      specified_drift_arrays=[point_log])

    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['point_log'], point_drift=well)
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)

    uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['specified'],
                               specified_drift=[self.simple_data[:, 0], self.simple_data[:, 1], point_log_data])
    z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                                      specified_drift_arrays=[xg, yg, point_log])
    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear', 'point_log'],
                              point_drift=well)
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)


def test_uk_functional_drift(self):

    well = np.array([[1.1, 1.1, -1.0]])
    func_x = lambda x, y: x
    func_y = lambda x, y: y
    func_well = lambda x, y: - well[0, 2] * np.log(np.sqrt((x - well[0, 0])**2. + (y - well[0, 1])**2.))

    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='linear', drift_terms=['functional'])
    with pytest.raises(TypeError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='linear', drift_terms=['functional'],
                         functional_drift=func_x)

    uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['functional'],
                               functional_drift=[func_x, func_y])
    z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear'])
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)

    uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['functional'], functional_drift=[func_well])
    z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['point_log'], point_drift=well)
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)

    uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                               variogram_model='linear', drift_terms=['functional'],
                               functional_drift=[func_x, func_y, func_well])
    z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
    uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear', 'point_log'],
                              point_drift=well)
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)


def test_uk_with_external_drift(self):

    dem, demx, demy, cellsize, no_data = \
        kt.read_asc_grid(os.path.join(BASE_DIR, 'test_data/test3_dem.asc'))
    uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                          variogram_model='spherical',
                          variogram_parameters=[500.0, 3000.0, 0.0],
                          anisotropy_scaling=1.0, anisotropy_angle=0.0,
                          drift_terms=['external_Z'], external_drift=dem,
                          external_drift_x=demx, external_drift_y=demy,
                          verbose=False)
    answer, gridx, gridy, cellsize, no_data = \
        kt.read_asc_grid(os.path.join(BASE_DIR, 'test_data/test3_answer.asc'))

    z, ss = uk.execute('grid', gridx, gridy, backend='vectorized')
    assert_allclose(z, answer, **allclose_pars)

    z, ss = uk.execute('grid', gridx, gridy, backend='loop')
    assert_allclose(z, answer, **allclose_pars)


def test_force_exact(self):
    data = np.array([[1., 1., 2.],
                     [2., 2., 1.5],
                     [3., 3., 1.]])
    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model='linear',
                         variogram_parameters=[1.0, 1.0])
    z, ss = ok.execute('grid', [1., 2., 3.], [1., 2., 3.],
                       backend='vectorized')
    self.assertAlmostEqual(z[0, 0], 2.0)
    self.assertAlmostEqual(ss[0, 0], 0.0)
    self.assertAlmostEqual(z[1, 1], 1.5)
    self.assertAlmostEqual(ss[1, 1], 0.0)
    self.assertAlmostEqual(z[2, 2], 1.0)
    self.assertAlmostEqual(ss[2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 0], 0.0)
    z, ss = ok.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.],
                       backend='vectorized')
    self.assertNotAlmostEqual(ss[0], 0.0)
    self.assertNotAlmostEqual(ss[1], 0.0)
    self.assertNotAlmostEqual(ss[2], 0.0)
    self.assertAlmostEqual(z[3], 1.0)
    self.assertAlmostEqual(ss[3], 0.0)
    z, ss = ok.execute('grid', np.arange(0., 4., 0.1),
                       np.arange(0., 4., 0.1), backend='vectorized')
    self.assertAlmostEqual(z[10, 10], 2.)
    self.assertAlmostEqual(ss[10, 10], 0.)
    self.assertAlmostEqual(z[20, 20], 1.5)
    self.assertAlmostEqual(ss[20, 20], 0.)
    self.assertAlmostEqual(z[30, 30], 1.0)
    self.assertAlmostEqual(ss[30, 30], 0.)
    self.assertNotAlmostEqual(ss[0, 0], 0.0)
    self.assertNotAlmostEqual(ss[15, 15], 0.0)
    self.assertNotAlmostEqual(ss[10, 0], 0.0)
    self.assertNotAlmostEqual(ss[0, 10], 0.0)
    self.assertNotAlmostEqual(ss[20, 10], 0.0)
    self.assertNotAlmostEqual(ss[10, 20], 0.0)
    self.assertNotAlmostEqual(ss[30, 20], 0.0)
    self.assertNotAlmostEqual(ss[20, 30], 0.0)
    z, ss = ok.execute('grid', np.arange(0., 3.1, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='vectorized')
    assert (np.any(ss <= 1e-15))
    assert not np.any(ss[:9, :30] <= 1e-15)
    assert not np.allclose(z[:9, :30], 0.)
    z, ss = ok.execute('grid', np.arange(0., 1.9, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='vectorized')
    assert not np.any(ss <= 1e-15)
    z, ss = ok.execute('masked', np.arange(2.5, 3.5, 0.1),
                       np.arange(2.5, 3.5, 0.25), backend='vectorized',
                       mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1),
                                       np.arange(2.5, 3.5, 0.25))[0] == 0.))
    assert (ss[2, 5] <= 1e-15)
    assert not np.allclose(ss, 0.)

    z, ss = ok.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='loop')
    self.assertAlmostEqual(z[0, 0], 2.0)
    self.assertAlmostEqual(ss[0, 0], 0.0)
    self.assertAlmostEqual(z[1, 1], 1.5)
    self.assertAlmostEqual(ss[1, 1], 0.0)
    self.assertAlmostEqual(z[2, 2], 1.0)
    self.assertAlmostEqual(ss[2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 0], 0.0)
    z, ss = ok.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.],
                       backend='loop')
    self.assertNotAlmostEqual(ss[0], 0.0)
    self.assertNotAlmostEqual(ss[1], 0.0)
    self.assertNotAlmostEqual(ss[2], 0.0)
    self.assertAlmostEqual(z[3], 1.0)
    self.assertAlmostEqual(ss[3], 0.0)
    z, ss = ok.execute('grid', np.arange(0., 4., 0.1),
                       np.arange(0., 4., 0.1), backend='loop')
    self.assertAlmostEqual(z[10, 10], 2.)
    self.assertAlmostEqual(ss[10, 10], 0.)
    self.assertAlmostEqual(z[20, 20], 1.5)
    self.assertAlmostEqual(ss[20, 20], 0.)
    self.assertAlmostEqual(z[30, 30], 1.0)
    self.assertAlmostEqual(ss[30, 30], 0.)
    self.assertNotAlmostEqual(ss[0, 0], 0.0)
    self.assertNotAlmostEqual(ss[15, 15], 0.0)
    self.assertNotAlmostEqual(ss[10, 0], 0.0)
    self.assertNotAlmostEqual(ss[0, 10], 0.0)
    self.assertNotAlmostEqual(ss[20, 10], 0.0)
    self.assertNotAlmostEqual(ss[10, 20], 0.0)
    self.assertNotAlmostEqual(ss[30, 20], 0.0)
    self.assertNotAlmostEqual(ss[20, 30], 0.0)
    z, ss = ok.execute('grid', np.arange(0., 3.1, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='loop')
    assert (np.any(ss <= 1e-15))
    assert not np.any(ss[:9, :30] <= 1e-15)
    assert not np.allclose(z[:9, :30], 0.)
    z, ss = ok.execute('grid', np.arange(0., 1.9, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='loop')
    assert not np.any(ss <= 1e-15)
    z, ss = ok.execute('masked', np.arange(2.5, 3.5, 0.1),
                       np.arange(2.5, 3.5, 0.25), backend='loop',
                       mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
    assert ss[2, 5] <= 1e-15
    assert not np.allclose(ss, 0.)

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2])
    z, ss = uk.execute('grid', [1., 2., 3.], [1., 2., 3.],
                       backend='vectorized')
    self.assertAlmostEqual(z[0, 0], 2.0)
    self.assertAlmostEqual(ss[0, 0], 0.0)
    self.assertAlmostEqual(z[1, 1], 1.5)
    self.assertAlmostEqual(ss[1, 1], 0.0)
    self.assertAlmostEqual(z[2, 2], 1.0)
    self.assertAlmostEqual(ss[2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 0], 0.0)
    z, ss = uk.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.],
                       backend='vectorized')
    self.assertNotAlmostEqual(ss[0], 0.0)
    self.assertNotAlmostEqual(ss[1], 0.0)
    self.assertNotAlmostEqual(ss[2], 0.0)
    self.assertAlmostEqual(z[3], 1.0)
    self.assertAlmostEqual(ss[3], 0.0)
    z, ss = uk.execute('grid', np.arange(0., 4., 0.1),
                       np.arange(0., 4., 0.1), backend='vectorized')
    self.assertAlmostEqual(z[10, 10], 2.)
    self.assertAlmostEqual(ss[10, 10], 0.)
    self.assertAlmostEqual(z[20, 20], 1.5)
    self.assertAlmostEqual(ss[20, 20], 0.)
    self.assertAlmostEqual(z[30, 30], 1.0)
    self.assertAlmostEqual(ss[30, 30], 0.)
    self.assertNotAlmostEqual(ss[0, 0], 0.0)
    self.assertNotAlmostEqual(ss[15, 15], 0.0)
    self.assertNotAlmostEqual(ss[10, 0], 0.0)
    self.assertNotAlmostEqual(ss[0, 10], 0.0)
    self.assertNotAlmostEqual(ss[20, 10], 0.0)
    self.assertNotAlmostEqual(ss[10, 20], 0.0)
    self.assertNotAlmostEqual(ss[30, 20], 0.0)
    self.assertNotAlmostEqual(ss[20, 30], 0.0)
    z, ss = uk.execute('grid', np.arange(0., 3.1, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='vectorized')
    assert np.any(ss <= 1e-15)
    assert not np.any(ss[:9, :30] <= 1e-15)
    assert not np.allclose(z[:9, :30], 0.)
    z, ss = uk.execute('grid', np.arange(0., 1.9, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='vectorized')
    assert not(np.any(ss <= 1e-15))
    z, ss = uk.execute('masked', np.arange(2.5, 3.5, 0.1),
                       np.arange(2.5, 3.5, 0.25), backend='vectorized',
                       mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
    assert (ss[2, 5] <= 1e-15)
    assert not np.allclose(ss, 0.)
    z, ss = uk.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='loop')
    self.assertAlmostEqual(z[0, 0], 2.0)
    self.assertAlmostEqual(ss[0, 0], 0.0)
    self.assertAlmostEqual(z[1, 1], 1.5)
    self.assertAlmostEqual(ss[1, 1], 0.0)
    self.assertAlmostEqual(z[2, 2], 1.0)
    self.assertAlmostEqual(ss[2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 0], 0.0)
    z, ss = uk.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.],
                       backend='loop')
    self.assertNotAlmostEqual(ss[0], 0.0)
    self.assertNotAlmostEqual(ss[1], 0.0)
    self.assertNotAlmostEqual(ss[2], 0.0)
    self.assertAlmostEqual(z[3], 1.0)
    self.assertAlmostEqual(ss[3], 0.0)
    z, ss = uk.execute('grid', np.arange(0., 4., 0.1),
                       np.arange(0., 4., 0.1), backend='loop')
    self.assertAlmostEqual(z[10, 10], 2.)
    self.assertAlmostEqual(ss[10, 10], 0.)
    self.assertAlmostEqual(z[20, 20], 1.5)
    self.assertAlmostEqual(ss[20, 20], 0.)
    self.assertAlmostEqual(z[30, 30], 1.0)
    self.assertAlmostEqual(ss[30, 30], 0.)
    self.assertNotAlmostEqual(ss[0, 0], 0.0)
    self.assertNotAlmostEqual(ss[15, 15], 0.0)
    self.assertNotAlmostEqual(ss[10, 0], 0.0)
    self.assertNotAlmostEqual(ss[0, 10], 0.0)
    self.assertNotAlmostEqual(ss[20, 10], 0.0)
    self.assertNotAlmostEqual(ss[10, 20], 0.0)
    self.assertNotAlmostEqual(ss[30, 20], 0.0)
    self.assertNotAlmostEqual(ss[20, 30], 0.0)
    z, ss = uk.execute('grid', np.arange(0., 3.1, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='loop')
    assert np.any(ss <= 1e-15)
    assert not np.any(ss[:9, :30] <= 1e-15)
    assert not np.allclose(z[:9, :30], 0.)
    z, ss = uk.execute('grid', np.arange(0., 1.9, 0.1),
                       np.arange(2.1, 3.1, 0.1), backend='loop')
    assert not np.any(ss <= 1e-15)
    z, ss = uk.execute('masked', np.arange(2.5, 3.5, 0.1),
                       np.arange(2.5, 3.5, 0.25), backend='loop',
                       mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1),
                                       np.arange(2.5, 3.5, 0.25))[0] == 0.))
    assert (ss[2, 5] <= 1e-15)
    assert not np.allclose(ss, 0.)

    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1])).T,
                        data[:, 2], np.array([1., 1.]),
                        variogram_models.linear_variogram_model, [1.0, 1.0],
                        'euclidean')
    self.assertAlmostEqual(z, 2.)
    self.assertAlmostEqual(ss, 0.)
    z, ss = core._krige(np.vstack((data[:, 0], data[:, 1])).T,
                        data[:, 2], np.array([1., 2.]),
                        variogram_models.linear_variogram_model, [1.0, 1.0],
                        'euclidean')
    self.assertNotAlmostEqual(ss, 0.)

    data = np.zeros((50, 3))
    x, y = np.meshgrid(np.arange(0., 10., 1.), np.arange(0., 10., 2.))
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(x) * np.ravel(y)
    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model='linear',
                         variogram_parameters=[100.0, 1.0])
    z, ss = ok.execute('grid', np.arange(0., 10., 1.),
                       np.arange(0., 10., 2.), backend='vectorized')
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0., **allclose_pars)
    z, ss = ok.execute('grid', np.arange(0.5, 10., 1.),
                       np.arange(0.5, 10., 2.), backend='vectorized')
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.)
    z, ss = ok.execute('grid', np.arange(0., 10., 1.),
                       np.arange(0., 10., 2.), backend='loop')
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0., **allclose_pars)
    z, ss = ok.execute('grid', np.arange(0.5, 10., 1.),
                       np.arange(0.5, 10., 2.), backend='loop')
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.)

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                          variogram_model='linear',
                          variogram_parameters=[100.0, 1.0])
    z, ss = uk.execute('grid', np.arange(0., 10., 1.),
                       np.arange(0., 10., 2.), backend='vectorized')
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0., **allclose_pars)
    z, ss = uk.execute('grid', np.arange(0.5, 10., 1.),
                       np.arange(0.5, 10., 2.), backend='vectorized')
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.)
    z, ss = uk.execute('grid', np.arange(0., 10., 1.),
                       np.arange(0., 10., 2.), backend='loop')
    assert_allclose(np.ravel(z), data[:, 2], **allclose_pars)
    assert_allclose(ss, 0., **allclose_pars)
    z, ss = uk.execute('grid', np.arange(0.5, 10., 1.),
                       np.arange(0.5, 10., 2.), backend='loop')
    assert not np.allclose(np.ravel(z), data[:, 2])
    assert not np.allclose(ss, 0.)


def test_custom_variogram(self):
    func = lambda params, dist: params[0] * np.log10(dist + params[1]) + params[2]

    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='mrow')
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='custom')
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='custom', variogram_function=0)
    with pytest.raises(ValueError):
        UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                         self.simple_data[:, 2], variogram_model='custom', variogram_function=func)
    uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                          variogram_model='custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
    self.assertAlmostEqual(uk.variogram_function([1., 1., 1.], 1.), 1.3010, 4)
    uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                          variogram_model='linear')
    uk.update_variogram_model('custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
    self.assertAlmostEqual(uk.variogram_function([1., 1., 1.], 1.), 1.3010, 4)

    with pytest.raises(ValueError):
        OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                        self.simple_data[:, 2], variogram_model='mrow')
    with pytest.raises(ValueError):
        OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                        self.simple_data[:, 2], variogram_model='custom')
    with pytest.raises(ValueError):
        OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                        self.simple_data[:, 2], variogram_model='custom', variogram_function=0)
    with pytest.raises(ValueError):
        OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1],
                        self.simple_data[:, 2], variogram_model='custom', variogram_function=func)
    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                         variogram_model='custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
    self.assertAlmostEqual(ok.variogram_function([1., 1., 1.], 1.), 1.3010, 4)
    ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                         variogram_model='linear')
    ok.update_variogram_model('custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
    self.assertAlmostEqual(ok.variogram_function([1., 1., 1.], 1.), 1.3010, 4)


def test_ok3d(self):

    # Test to compare K3D results to those obtained using KT3D_H2O.
    # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)
    k3d = OrdinaryKriging3D(self.test_data[:, 0], self.test_data[:, 1], np.zeros(self.test_data[:, 1].shape),
                            self.test_data[:, 2], variogram_model='exponential',
                            variogram_parameters=[500.0, 3000.0, 0.0])
    k, ss = k3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='vectorized')
    assert_allclose(np.squeeze(k), self.ok_test_answer)
    k, ss = k3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='loop')
    assert_allclose(np.squeeze(k), self.ok_test_answer)

    # Test to compare K3D results to those obtained using KT3D.
    data = np.genfromtxt(os.path.join(BASE_DIR, 'test_data', 'test3d_data.txt'), skip_header=1)
    ans = np.genfromtxt(os.path.join(BASE_DIR, 'test_data', 'test3d_answer.txt'))
    ans_z = ans[:, 0].reshape((10, 10, 10))
    ans_ss = ans[:, 1].reshape((10, 10, 10))
    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                            variogram_model='linear', variogram_parameters=[1., 0.1])
    k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
    assert_allclose(k, ans_z, rtol=1e-3, atol=1e-8)
    assert_allclose(ss, ans_ss, rtol=1e-3, atol=1e-8)
    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                            variogram_model='linear', variogram_parameters=[1., 0.1])
    k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')
    assert_allclose(k, ans_z, rtol=1e-3, atol=1e-8)
    assert_allclose(ss, ans_ss, rtol=1e-3, atol=1e-8)


def test_ok3d_moving_window(self):

    # Test to compare K3D results to those obtained using KT3D.
    data = np.genfromtxt(os.path.join(BASE_DIR, 'test_data', 'test3d_data.txt'),
                         skip_header=1)
    ans = np.genfromtxt(os.path.join(BASE_DIR, './test_data/test3d_answer.txt'))
    ans_z = ans[:, 0].reshape((10, 10, 10))
    ans_ss = ans[:, 1].reshape((10, 10, 10))
    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                            variogram_model='linear', variogram_parameters=[1., 0.1])
    k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop', n_closest_points=10)
    assert_allclose(k, ans_z, rtol=1e-3)
    assert_allclose(ss, ans_ss, rtol=1e-3)


def test_ok3d_uk3d_and_backends_produce_same_results(self):

    ok3d = OrdinaryKriging3D(self.test_data[:, 0], self.test_data[:, 1], np.zeros(self.test_data[:, 1].shape),
                             self.test_data[:, 2], variogram_model='exponential',
                             variogram_parameters=[500.0, 3000.0, 0.0])
    ok_v, oss_v = ok3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='vectorized')
    ok_l, oss_l = ok3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='loop')

    uk3d = UniversalKriging3D(self.test_data[:, 0], self.test_data[:, 1], np.zeros(self.test_data[:, 1].shape),
                              self.test_data[:, 2], variogram_model='exponential',
                              variogram_parameters=[500., 3000., 0.])
    uk_v, uss_v = uk3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='vectorized')
    assert_allclose(uk_v, ok_v)
    uk_l, uss_l = uk3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='loop')
    assert_allclose(uk_l, ok_l)
    assert_allclose(uk_l, uk_v)
    assert_allclose(uss_l, uss_v)

    data = np.genfromtxt(os.path.join(BASE_DIR, 'test_data', 'test3d_data.txt'), skip_header=1)
    ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                             variogram_model='linear', variogram_parameters=[1., 0.1])
    ok_v, oss_v = ok3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
    ok_l, oss_l = ok3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')

    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                              variogram_model='linear', variogram_parameters=[1., 0.1])
    uk_v, uss_v = uk3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
    assert_allclose(uk_v, ok_v)
    assert_allclose(uss_v, oss_v)
    uk_l, uss_l = uk3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')
    assert_allclose(uk_l, ok_l)
    assert_allclose(uss_l, oss_l)
    assert_allclose(uk_l, uk_v)
    assert_allclose(uss_l, uss_v)


def test_ok3d_update_variogram_model(self):

    with pytest.raises(ValueError):
        OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                          self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='blurg')

    k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                            self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])
    variogram_model = k3d.variogram_model
    variogram_parameters = k3d.variogram_model_parameters
    anisotropy_scaling_y = k3d.anisotropy_scaling_y
    anisotropy_scaling_z = k3d.anisotropy_scaling_z
    anisotropy_angle_x = k3d.anisotropy_angle_x
    anisotropy_angle_y = k3d.anisotropy_angle_y
    anisotropy_angle_z = k3d.anisotropy_angle_z

    with pytest.raises(ValueError):
        k3d.update_variogram_model('blurg')
    k3d.update_variogram_model('power', anisotropy_scaling_y=3.0, anisotropy_scaling_z=3.0,
                               anisotropy_angle_x=45.0, anisotropy_angle_y=45.0, anisotropy_angle_z=45.0)
    assert variogram_model != k3d.variogram_model
    assert variogram_parameters != k3d.variogram_model_parameters
    assert anisotropy_scaling_y != k3d.anisotropy_scaling_y
    assert anisotropy_scaling_z != k3d.anisotropy_scaling_z
    assert anisotropy_angle_x != k3d.anisotropy_angle_x
    assert anisotropy_angle_y != k3d.anisotropy_angle_y
    assert anisotropy_angle_z != k3d.anisotropy_angle_z


def test_uk3d_update_variogram_model(self):

    with pytest.raises(ValueError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                           self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='blurg')

    uk3d = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                              self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])
    variogram_model = uk3d.variogram_model
    variogram_parameters = uk3d.variogram_model_parameters
    anisotropy_scaling_y = uk3d.anisotropy_scaling_y
    anisotropy_scaling_z = uk3d.anisotropy_scaling_z
    anisotropy_angle_x = uk3d.anisotropy_angle_x
    anisotropy_angle_y = uk3d.anisotropy_angle_y
    anisotropy_angle_z = uk3d.anisotropy_angle_z

    with pytest.raises(ValueError):
        uk3d.update_variogram_model('blurg')
    uk3d.update_variogram_model('power', anisotropy_scaling_y=3.0, anisotropy_scaling_z=3.0,
                                anisotropy_angle_x=45.0, anisotropy_angle_y=45.0, anisotropy_angle_z=45.0)
    assert not variogram_model == uk3d.variogram_model
    assert not variogram_parameters == uk3d.variogram_model_parameters
    assert not anisotropy_scaling_y == uk3d.anisotropy_scaling_y
    assert not anisotropy_scaling_z == uk3d.anisotropy_scaling_z
    assert not anisotropy_angle_x == uk3d.anisotropy_angle_x
    assert not anisotropy_angle_y == uk3d.anisotropy_angle_y
    assert not anisotropy_angle_z == uk3d.anisotropy_angle_z

def test_ok3d_backends_produce_same_result(self):

    k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                            self.simple_data_3d[:, 3], variogram_model='linear')
    k_k3d_v, ss_k3d_v = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                    backend='vectorized')
    k_k3d_l, ss_k3d_l = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                    backend='loop')
    assert_allclose(k_k3d_v, k_k3d_l)
    assert_allclose(ss_k3d_v, ss_k3d_l)


def test_ok3d_execute(self):

    k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                            self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])

    with pytest.raises(ValueError):
        k3d.execute('blurg', self.simple_gridx_3d,
                    self.simple_gridy_3d, self.simple_gridz_3d)

    k, ss = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                        self.simple_gridz_3d, backend='vectorized')
    shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    k, ss = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                        self.simple_gridz_3d, backend='loop')
    shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    with pytest.raises(IOError):
        k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                    self.simple_gridz_3d, backend='vectorized')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                    self.simple_gridz_3d, mask=mask, backend='vectorized')
    k, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                        mask=self.mask_3d, backend='vectorized')
    assert (np.ma.is_masked(k))
    assert (np.ma.is_masked(ss))
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                        mask=self.mask_3d.T, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(IOError):
        k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                    self.simple_gridz_3d, backend='loop')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                    self.simple_gridz_3d, mask=mask, backend='loop')
    k, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                        mask=self.mask_3d, backend='loop')
    assert (np.ma.is_masked(k))
    assert (np.ma.is_masked(ss))
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                        mask=self.mask_3d.T, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        k3d.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                    np.array([1.0]), backend='vectorized')
    k, ss = k3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                        self.simple_gridz_3d[0], backend='vectorized')
    assert k.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        k3d.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                    np.array([1.0]), backend='loop')
    k, ss = k3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                        self.simple_gridz_3d[0], backend='loop')
    assert k.shape == (1,)
    assert ss.shape == (1,)

    data = np.zeros((125, 4))
    z, y, x = np.meshgrid(np.arange(0., 5., 1.), np.arange(0., 5., 1.), np.arange(0., 5., 1.))
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(z)
    data[:, 3] = np.ravel(z)
    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
    k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                        np.arange(0., 4., 1.), backend='vectorized')
    assert_allclose(k[0, :, :], 0., atol=0.01)
    assert_allclose(k[1, :, :], 1., rtol=1.e-2)
    assert_allclose(k[2, :, :], 2., rtol=1.e-2)
    assert_allclose(k[3, :, :], 3., rtol=1.e-2)
    k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                        np.arange(0., 4., 1.), backend='loop')
    assert_allclose(k[0, :, :], 0., atol=0.01)
    assert_allclose(k[1, :, :], 1., rtol=1.e-2)
    assert_allclose(k[2, :, :], 2., rtol=1.e-2)
    assert_allclose(k[3, :, :], 3., rtol=1.e-2)
    k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
    k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='vectorized')
    assert_allclose(k[0], 1., atol=0.01)
    assert_allclose(k[1], 2., rtol=1.e-2)
    assert_allclose(k[2], 3., rtol=1.e-2)
    k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='loop')
    assert_allclose(k[0], 1., atol=0.01)
    assert_allclose(k[1], 2., rtol=1.e-2)
    assert_allclose(k[2], 3., rtol=1.e-2)


def test_uk3d_execute(self):

    uk3d = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                              self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])

    with pytest.raises(ValueError):
        uk3d.execute('blurg', self.simple_gridx_3d,
                     self.simple_gridy_3d, self.simple_gridz_3d)

    k, ss = uk3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                         self.simple_gridz_3d, backend='vectorized')
    shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    k, ss = uk3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                         self.simple_gridz_3d, backend='loop')
    shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
    assert k.shape == shape
    assert ss.shape == shape
    assert np.amax(k) != np.amin(k)
    assert np.amax(ss) != np.amin(ss)
    assert not np.ma.is_masked(k)

    with pytest.raises(IOError):
        uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                     self.simple_gridz_3d, backend='vectorized')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                     self.simple_gridz_3d, mask=mask, backend='vectorized')
    k, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                         mask=self.mask_3d, backend='vectorized')
    assert (np.ma.is_masked(k))
    assert (np.ma.is_masked(ss))
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                         mask=self.mask_3d.T, backend='vectorized')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(IOError):
        uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                     self.simple_gridz_3d, backend='loop')
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d,
                     self.simple_gridz_3d, mask=mask, backend='loop')
    k, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                         mask=self.mask_3d, backend='loop')
    assert (np.ma.is_masked(k))
    assert (np.ma.is_masked(ss))
    assert k[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked
    z, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                         mask=self.mask_3d.T, backend='loop')
    assert (np.ma.is_masked(z))
    assert (np.ma.is_masked(ss))
    assert z[0, 0, 0] is np.ma.masked
    assert ss[0, 0, 0] is np.ma.masked

    with pytest.raises(ValueError):
        uk3d.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                     np.array([1.0]), backend='vectorized')
    k, ss = uk3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                         self.simple_gridz_3d[0], backend='vectorized')
    assert k.shape == (1,)
    assert ss.shape == (1,)

    with pytest.raises(ValueError):
        uk3d.execute('points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                     np.array([1.0]), backend='loop')
    k, ss = uk3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                         self.simple_gridz_3d[0], backend='loop')
    assert k.shape == (1,)
    assert ss.shape == (1,)

    data = np.zeros((125, 4))
    z, y, x = np.meshgrid(np.arange(0., 5., 1.), np.arange(0., 5., 1.), np.arange(0., 5., 1.))
    data[:, 0] = np.ravel(x)
    data[:, 1] = np.ravel(y)
    data[:, 2] = np.ravel(z)
    data[:, 3] = np.ravel(z)
    k3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
    k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                        np.arange(0., 4., 1.), backend='vectorized')
    assert_allclose(k[0, :, :], 0., atol=0.01)
    assert_allclose(k[1, :, :], 1., rtol=1.e-2)
    assert_allclose(k[2, :, :], 2., rtol=1.e-2)
    assert_allclose(k[3, :, :], 3., rtol=1.e-2)
    k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                        np.arange(0., 4., 1.), backend='loop')
    assert_allclose(k[0, :, :], 0., atol=0.01)
    assert_allclose(k[1, :, :], 1., rtol=1.e-2)
    assert_allclose(k[2, :, :], 2., rtol=1.e-2)
    assert_allclose(k[3, :, :], 3., rtol=1.e-2)
    k3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
    k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='vectorized')
    assert_allclose(k[0], 1., atol=0.01)
    assert_allclose(k[1], 2., rtol=1.e-2)
    assert_allclose(k[2], 3., rtol=1.e-2)
    k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='loop')
    assert_allclose(k[0], 1., atol=0.01)
    assert_allclose(k[1], 2., rtol=1.e-2)
    assert_allclose(k[2], 3., rtol=1.e-2)


def test_force_exact_3d(self):
    k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                            self.simple_data_3d[:, 3], variogram_model='linear')
    k, ss = k3d.execute('grid', [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend='vectorized')
    self.assertAlmostEqual(k[2, 0, 0], 0.9)
    self.assertAlmostEqual(ss[2, 0, 0], 0.0)
    self.assertAlmostEqual(k[0, 2, 0], 0.9)
    self.assertAlmostEqual(ss[0, 2, 0], 0.0)
    self.assertAlmostEqual(k[1, 2, 2], 0.7)
    self.assertAlmostEqual(ss[1, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 0, 0], 0.0)

    k, ss = k3d.execute('grid', [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend='loop')
    self.assertAlmostEqual(k[2, 0, 0], 0.9)
    self.assertAlmostEqual(ss[2, 0, 0], 0.0)
    self.assertAlmostEqual(k[0, 2, 0], 0.9)
    self.assertAlmostEqual(ss[0, 2, 0], 0.0)
    self.assertAlmostEqual(k[1, 2, 2], 0.7)
    self.assertAlmostEqual(ss[1, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 0, 0], 0.0)

    k3d = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                             self.simple_data_3d[:, 3], variogram_model='linear')
    k, ss = k3d.execute('grid', [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend='vectorized')
    self.assertAlmostEqual(k[2, 0, 0], 0.9)
    self.assertAlmostEqual(ss[2, 0, 0], 0.0)
    self.assertAlmostEqual(k[0, 2, 0], 0.9)
    self.assertAlmostEqual(ss[0, 2, 0], 0.0)
    self.assertAlmostEqual(k[1, 2, 2], 0.7)
    self.assertAlmostEqual(ss[1, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 0, 0], 0.0)

    k, ss = k3d.execute('grid', [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], backend='loop')
    self.assertAlmostEqual(k[2, 0, 0], 0.9)
    self.assertAlmostEqual(ss[2, 0, 0], 0.0)
    self.assertAlmostEqual(k[0, 2, 0], 0.9)
    self.assertAlmostEqual(ss[0, 2, 0], 0.0)
    self.assertAlmostEqual(k[1, 2, 2], 0.7)
    self.assertAlmostEqual(ss[1, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[2, 2, 2], 0.0)
    self.assertNotAlmostEqual(ss[0, 0, 0], 0.0)


def test_uk3d_specified_drift(self):

    zg, yg, xg = np.meshgrid(self.simple_gridz_3d, self.simple_gridy_3d, self.simple_gridx_3d, indexing='ij')

    with pytest.raises(ValueError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                      self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                      variogram_model='linear', drift_terms=['specified'])
    with pytest.raises(TypeError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                      self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='linear',
                      drift_terms=['specified'], specified_drift=self.simple_data_3d[:, 0])
    with pytest.raises(ValueError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                      self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='linear',
                      drift_terms=['specified'], specified_drift=[self.simple_data_3d[:2, 0]])

    uk_spec = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                 self.simple_data_3d[:, 3], variogram_model='linear', drift_terms=['specified'],
                                 specified_drift=[self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                                                  self.simple_data_3d[:, 2]])
    with pytest.raises(ValueError):
        uk_spec.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                      self.simple_gridz_3d, specified_drift_arrays=[self.simple_gridx_3d, self.simple_gridy_3d,
                                                                    self.simple_gridz_3d])
    with pytest.raises(TypeError):
        uk_spec.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                      self.simple_gridz_3d, specified_drift_arrays=self.simple_gridx_3d)
    with pytest.raises(ValueError):
        uk_spec.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                      self.simple_gridz_3d, specified_drift_arrays=[zg])
    z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                      specified_drift_arrays=[xg, yg, zg])

    uk_lin = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                self.simple_data_3d[:, 3], variogram_model='linear',
                                drift_terms=['regional_linear'])
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d)

    assert_allclose(z_spec, z_lin)
    assert_allclose(ss_spec, ss_lin)


def test_uk3d_functional_drift(self):

    func_x = lambda x, y, z: x
    func_y = lambda x, y, z: y
    func_z = lambda x, y, z: z

    with pytest.raises(ValueError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                           self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                           variogram_model='linear', drift_terms=['functional'])
    with pytest.raises(TypeError):
        UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                           self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='linear',
                           drift_terms=['functional'], functional_drift=func_x)

    uk_func = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                 self.simple_data_3d[:, 3], variogram_model='linear', drift_terms=['functional'],
                                 functional_drift=[func_x, func_y, func_z])
    z_func, ss_func = uk_func.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d)
    uk_lin = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                self.simple_data_3d[:, 3], variogram_model='linear',
                                drift_terms=['regional_linear'])
    z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d)
    assert_allclose(z_func, z_lin)
    assert_allclose(ss_func, ss_lin)


def test_geometric_code(self):

    # Create selected points distributed across the sphere:
    N = 4
    lon = np.array([7.0,   7.0,     187.0,  73.231])
    lat = np.array([13.23, 13.2301, -13.23, -79.3])

    # For the points generated with this reference seed, the distance matrix
    # has been calculated using geopy (v. 1.11.0) as follows:
    # >>>   from geopy.distance import great_circle
    # >>>   g = great_circle(radius=1.0)
    # >>>   d = np.zeros((N,N), dtype=float)
    # >>>   for i in range(N):
    # >>>       for j in range(N):
    # >>>           d[i,j] = g.measure((lat[i],lon[i]),(lat[j],lon[j]))
    # >>>   d *= 180.0/np.pi
    # From that distance matrix, the reference values have been obtained.
    d_ref = np.array(
            [[0.0, 1e-4, 180.0, 98.744848317171801],
             [1e-4, 0.0, 179.9999, 98.744946828324345],
             [180.0, 179.9999, 0.0, 81.255151682828213],
             [98.744848317171801, 98.744946828324345, 81.255151682828213, 0.0]]
            )
    
    # Calculate distance matrix using the PyKrige code:
    d = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            d[i,j] = core.great_circle_distance(lon[i],lat[i],lon[j],lat[j])
    
    # Test agains reference values:
    
    np.testing.assert_allclose(d, d_ref)
    
    # Test general features:
    np.testing.assert_allclose(d[np.eye(N,dtype=bool)], 0.0)
    np.testing.assert_equal(d>=0.0, np.ones((N,N),dtype=bool))
    np.testing.assert_allclose(d,d.T)
    np.testing.assert_equal(d<=180.0,np.ones((N,N),dtype=bool))

    # Test great_circle_distance and euclid3_to_great_circle against each other:
    lon_ref = lon
    lat_ref = lat
    for i in range(len(lon_ref)):
        lon, lat = np.meshgrid(np.linspace(0, 360.0, 20),
                               np.linspace(-90.0, 90.0, 20))
        dx = np.cos(np.pi/180.0*lon)*np.cos(np.pi/180.0*lat)- \
             np.cos(np.pi/180.0*lon_ref[i])*np.cos(np.pi/180.0*lat_ref[i])
        dy = np.sin(np.pi/180.0*lon)*np.cos(np.pi/180.0*lat)- \
             np.sin(np.pi/180.0*lon_ref[i])*np.cos(np.pi/180.0*lat_ref[i])
        dz = np.sin(np.pi/180.0*lat) - np.sin(np.pi/180.0*lat_ref[i])
        np.testing.assert_allclose(core.great_circle_distance(lon_ref[i], lat_ref[i], lon, lat),
            core.euclid3_to_great_circle(np.sqrt(dx**2+dy**2+dz**2)), rtol=1e-5)


def test_ok_geometric(self):
    # Generate random data:
    np.random.seed(89239413)
    lon = 360.0*np.random.rand(50, 1)
    lat = 180.0*np.random.rand(50, 1) - 90.0
    z = np.random.rand(50, 1)
    #data = np.concatenate((lon, lat, z), 1)

    # Generate grid:
    grid_lon = 360.0*np.random.rand(120, 1)
    grid_lat = 180.0*np.random.rand(120, 1) - 90.0

    # Create ordinary kriging object:
    OK = OrdinaryKriging(lon, lat, z, variogram_model='linear', verbose=False,
                         enable_plotting=False, coordinates_type='geographic')

    # Execute on grid:
    z, ss = OK.execute('grid', grid_lon, grid_lat)
