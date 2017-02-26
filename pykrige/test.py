from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""
Testing code.
Updated BSM March 2016
"""

import unittest
import os
import numpy as np
from itertools import product

from pykrige import kriging_tools as kt
from pykrige import core
from pykrige import variogram_models
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from pykrige.compat import SKLEARN_INSTALLED


class TestPyKrige(unittest.TestCase):

    def setUp(self):

        self.test_data = np.genfromtxt(os.path.join(os.getcwd(), 'test_data/test_data.txt'))
        self.ok_test_answer, self.ok_test_gridx, self.ok_test_gridy, cellsize, no_data = \
            kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/test1_answer.asc'), footer=2)
        self.uk_test_answer, self.uk_test_gridx, self.uk_test_gridy, cellsize, no_data = \
            kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/test2_answer.asc'), footer=2)

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
        zi, yi, xi = np.meshgrid(self.simple_gridz_3d, self.simple_gridy_3d, self.simple_gridx_3d, indexing='ij')
        self.mask_3d = np.array((xi == yi) & (yi == zi))

    def test_core_adjust_for_anisotropy(self):

        X = np.array([[1.0, 0.0, -1.0, 0.0],
                      [0.0, 1.0, 0.0, -1.0]]).T
        X_adj = core._adjust_for_anisotropy(X, [0.0, 0.0], [2.0], [90.0])
        self.assertTrue(np.allclose(X_adj[:, 0], np.array([0.0, 1.0, 0.0, -1.0])))
        self.assertTrue(np.allclose(X_adj[:, 1], np.array([-2.0, 0.0, 2.0, 0.0])))

    def test_core_adjust_for_anisotropy_3d(self):

        # this is a bad examples, as the X matrix is symmetric
        # and insensitive to transpositions
        X = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]).T
        X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [90., 0., 0.])
        self.assertTrue(np.allclose(X_adj[:, 0], np.array([1., 0., 0.])))
        self.assertTrue(np.allclose(X_adj[:, 1], np.array([0., 0., 2.])))
        self.assertTrue(np.allclose(X_adj[:, 2], np.array([0., -2., 0.])))
        X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [0., 90., 0.])
        self.assertTrue(np.allclose(X_adj[:, 0], np.array([0., 0., -1.])))
        self.assertTrue(np.allclose(X_adj[:, 1], np.array([0., 2., 0.])))
        self.assertTrue(np.allclose(X_adj[:, 2], np.array([2., 0., 0.])))
        X_adj = core._adjust_for_anisotropy(X, [0., 0., 0.], [2., 2.], [0., 0., 90.])
        self.assertTrue(np.allclose(X_adj[:, 0], np.array([0., 1., 0.])))
        self.assertTrue(np.allclose(X_adj[:, 1], np.array([-2., 0., 0.])))
        self.assertTrue(np.allclose(X_adj[:, 2], np.array([0., 0., 2.])))

    def test_core_initialize_variogram_model(self):

        # Note the variogram_function argument is not a string in real life...
        self.assertRaises(ValueError, core.initialize_variogram_model, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], 'linear', [0.0], 'linear', 6, False,
                          'euclidean')

        self.assertRaises(ValueError, core.initialize_variogram_model, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], 'spherical', [0.0], 'spherical', 6, False,
                          'euclidean')

        x = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
        y = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
        z = np.arange(1.0, 5.0, 1.0)
        lags, semivariance, variogram_model_parameters = core.initialize_variogram_model(x, y, z, 'linear',
                                                                                         [0.0, 0.0], 'linear',
                                                                                         6, False, 'euclidean')

        self.assertTrue(np.allclose(lags, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(semivariance, np.array([0.5, 2.0, 4.5])))

    def test_core_initialize_variogram_model_3d(self):

        # Note the variogram_function argument is not a string in real life...
        self.assertRaises(ValueError, core.initialize_variogram_model_3d, self.simple_data_3d[:, 0],
                          self.simple_data_3d[:, 1], self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                          'linear', [0.0], 'linear', 6, False)

        self.assertRaises(ValueError, core.initialize_variogram_model_3d, self.simple_data_3d[:, 0],
                          self.simple_data_3d[:, 1], self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                          'spherical', [0.0], 'spherical', 6, False)

        lags, semivariance, variogram_model_parameters = core.initialize_variogram_model_3d(np.array([1., 2., 3., 4.]),
                                                                                            np.array([1., 2., 3., 4.]),
                                                                                            np.array([1., 2., 3., 4.]),
                                                                                            np.array([1., 2., 3., 4.]),
                                                                                            'linear', [0.0, 0.0],
                                                                                            'linear', 3, False)
        self.assertTrue(np.allclose(lags, np.array([np.sqrt(3.), 2.*np.sqrt(3.), 3.*np.sqrt(3.)])))
        self.assertTrue(np.allclose(semivariance, np.array([0.5, 2.0, 4.5])))

    def test_core_calculate_variogram_model(self):

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.05, 2.95, 4.05, 4.95]),
                                             'linear', variogram_models.linear_variogram_model, False)
        self.assertTrue(np.allclose(res, np.array([0.98, 1.05]), 0.01, 0.01))

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.05, 2.95, 4.05, 4.95]),
                                             'linear', variogram_models.linear_variogram_model, True)
        self.assertTrue(np.allclose(res, np.array([0.98, 1.05]), 0.01, 0.01))

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.8284, 5.1962, 8.0]),
                                             'power', variogram_models.power_variogram_model, False)
        self.assertTrue(np.allclose(res, np.array([1.0, 1.5, 0.0])))

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.4142, 1.7321, 2.0]),
                                             'power', variogram_models.power_variogram_model, False)
        self.assertTrue(np.allclose(res, np.array([1.0, 0.5, 0.0])))

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.2642, 1.7293, 1.9004, 1.9634]),
                                             'exponential', variogram_models.exponential_variogram_model, False)
        self.assertTrue(np.allclose(res, np.array([2.0, 3.0, 0.0]), 0.001, 0.001))

        res = core.calculate_variogram_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5769, 1.4872, 1.9065, 1.9914]),
                                             'gaussian', variogram_models.gaussian_variogram_model, False)
        self.assertTrue(np.allclose(res, np.array([2.0, 3.0, 0.0]), 0.001, 0.001))

    def test_core_krige(self):

        # Example 3.2 from Kitanidis
        data = np.array([[9.7, 47.6, 1.22],
                         [43.8, 24.6, 2.822]])
        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (18.8, 67.9),
                           variogram_models.linear_variogram_model, [0.006, 0.1],
                           'euclidean')
        self.assertAlmostEqual(z, 1.6364, 4)
        self.assertAlmostEqual(ss, 0.4201, 4)

        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (43.8, 24.6),
                           variogram_models.linear_variogram_model, [0.006, 0.1],
                           'euclidean')
        self.assertAlmostEqual(z, 2.822, 3)
        self.assertAlmostEqual(ss, 0.0, 3)

    def test_core_krige_3d(self):

        # Adapted from example 3.2 from Kitanidis
        data = np.array([[9.7, 47.6, 1.0, 1.22],
                         [43.8, 24.6, 1.0, 2.822]])
        z, ss = core.krige_3d(data[:, 0], data[:, 1], data[:, 2], data[:, 3], (18.8, 67.9, 1.0),
                              variogram_models.linear_variogram_model, [0.006, 0.1])
        self.assertAlmostEqual(z, 1.6364, 4)
        self.assertAlmostEqual(ss, 0.4201, 4)

        z, ss = core.krige_3d(data[:, 0], data[:, 1], data[:, 2], data[:, 3], (43.8, 24.6, 1.0),
                              variogram_models.linear_variogram_model, [0.006, 0.1])
        self.assertAlmostEqual(z, 2.822, 3)
        self.assertAlmostEqual(ss, 0.0, 3)

    def test_ok(self):

        # Test to compare OK results to those obtained using KT3D_H2O.
        # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                             variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0])
        z, ss = ok.execute('grid', self.ok_test_gridx, self.ok_test_gridy, backend='vectorized')
        self.assertTrue(np.allclose(z, self.ok_test_answer))
        z, ss = ok.execute('grid', self.ok_test_gridx, self.ok_test_gridy, backend='loop')
        self.assertTrue(np.allclose(z, self.ok_test_answer))

    def test_ok_update_variogram_model(self):

        self.assertRaises(ValueError, OrdinaryKriging, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], variogram_model='blurg')

        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2])
        variogram_model = ok.variogram_model
        variogram_parameters = ok.variogram_model_parameters
        anisotropy_scaling = ok.anisotropy_scaling
        anisotropy_angle = ok.anisotropy_angle

        self.assertRaises(ValueError, ok.update_variogram_model, 'blurg')
        ok.update_variogram_model('power', anisotropy_scaling=3.0, anisotropy_angle=45.0)
        self.assertFalse(variogram_model == ok.variogram_model)
        self.assertFalse(variogram_parameters == ok.variogram_model_parameters)
        self.assertFalse(anisotropy_scaling == ok.anisotropy_scaling)
        self.assertFalse(anisotropy_angle == ok.anisotropy_angle)

    def test_ok_execute(self):

        ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])

        self.assertRaises(ValueError, ok.execute, 'blurg', self.simple_gridx, self.simple_gridy)

        z, ss = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='vectorized')
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        z, ss = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        self.assertRaises(IOError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy, backend='vectorized')
        mask = np.array([True, False])
        self.assertRaises(ValueError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask,
                          backend='vectorized')
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(IOError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy, backend='loop')
        mask = np.array([True, False])
        self.assertRaises(ValueError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask,
                          backend='loop')
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(ValueError, ok.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          backend='vectorized')
        z, ss = ok.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='vectorized')
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        self.assertRaises(ValueError, ok.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          backend='loop')
        z, ss = ok.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='loop')
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

    def test_cython_ok(self):
        ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
        z1, ss1 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
        z2, ss2 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='C')
        self.assertTrue(np.allclose(z1, z2))
        self.assertTrue(np.allclose(ss1, ss2))

        closest_points = 4

        z1, ss1 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop',
                             n_closest_points=closest_points)
        z2, ss2 = ok.execute('grid', self.simple_gridx, self.simple_gridy, backend='C',
                             n_closest_points=closest_points)
        self.assertTrue(np.allclose(z1, z2))
        self.assertTrue(np.allclose(ss1, ss2))

    def test_uk(self):

        # Test to compare UK with linear drift to results from KT3D_H2O.
        # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0],
                              drift_terms=['regional_linear'])
        z, ss = uk.execute('grid', self.uk_test_gridx, self.uk_test_gridy, backend='vectorized')
        self.assertTrue(np.allclose(z, self.uk_test_answer))
        z, ss = uk.execute('grid', self.uk_test_gridx, self.uk_test_gridy, backend='loop')
        self.assertTrue(np.allclose(z, self.uk_test_answer))

    def test_uk_update_variogram_model(self):

        self.assertRaises(ValueError, UniversalKriging, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], variogram_model='blurg')
        self.assertRaises(ValueError, UniversalKriging, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], drift_terms=['external_Z'])
        self.assertRaises(ValueError, UniversalKriging, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], drift_terms=['external_Z'], external_drift=np.array([0]))
        self.assertRaises(ValueError, UniversalKriging, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], drift_terms=['point_log'])

        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2])
        variogram_model = uk.variogram_model
        variogram_parameters = uk.variogram_model_parameters
        anisotropy_scaling = uk.anisotropy_scaling
        anisotropy_angle = uk.anisotropy_angle

        self.assertRaises(ValueError, uk.update_variogram_model, 'blurg')
        uk.update_variogram_model('power', anisotropy_scaling=3.0, anisotropy_angle=45.0)
        self.assertFalse(variogram_model == uk.variogram_model)
        self.assertFalse(variogram_parameters == uk.variogram_model_parameters)
        self.assertFalse(anisotropy_scaling == uk.anisotropy_scaling)
        self.assertFalse(anisotropy_angle == uk.anisotropy_angle)

    def test_uk_calculate_data_point_zscalars(self):

        dem = np.arange(0.0, 5.1, 0.1)
        dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
        dem_x = np.arange(0.0, 5.1, 0.1)
        dem_y = np.arange(0.0, 6.0, 1.0)

        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                          drift_terms=['external_Z'])
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                          drift_terms=['external_Z'], external_drift=dem)
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', variogram_parameters=[1.0, 0.0],
                          drift_terms=['external_Z'], external_drift=dem, external_drift_x=dem_x,
                          external_drift_y=np.arange(0.0, 5.0, 1.0))

        uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', variogram_parameters=[1.0, 0.0],
                              drift_terms=['external_Z'], external_drift=dem, external_drift_x=dem_x,
                              external_drift_y=dem_y)
        self.assertTrue(np.allclose(uk.z_scalars, self.simple_data[:, 0]))

        xi, yi = np.meshgrid(np.arange(0.0, 5.3, 0.1), self.simple_gridy)
        self.assertRaises(ValueError, uk._calculate_data_point_zscalars, xi, yi)

        xi, yi = np.meshgrid(np.arange(0.0, 5.0, 0.1), self.simple_gridy)
        z_scalars = uk._calculate_data_point_zscalars(xi, yi)
        self.assertTrue(np.allclose(z_scalars[0, :], np.arange(0.0, 5.0, 0.1)))

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

        self.assertRaises(ValueError, uk.execute, 'blurg', self.simple_gridx, self.simple_gridy)
        self.assertRaises(ValueError, uk.execute, 'grid', self.simple_gridx, self.simple_gridy, backend='mrow')

        z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='vectorized')
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        self.assertRaises(IOError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy, backend='vectorized')
        mask = np.array([True, False])
        self.assertRaises(ValueError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask,
                          backend='vectorized')
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(IOError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy, backend='loop')
        mask = np.array([True, False])
        self.assertRaises(ValueError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask,
                          backend='loop')
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(ValueError, uk.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          backend='vectorized')
        z, ss = uk.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='vectorized')
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        self.assertRaises(ValueError, uk.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          backend='loop')
        z, ss = uk.execute('points', self.simple_gridx[0], self.simple_gridy[0], backend='loop')
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

    def test_ok_uk_produce_same_result(self):

        gridx = np.linspace(1067000.0, 1072000.0, 100)
        gridy = np.linspace(241500.0, 244000.0, 100)
        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                             variogram_model='linear', verbose=False, enable_plotting=False)
        z_ok, ss_ok = ok.execute('grid', gridx, gridy, backend='vectorized')
        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='linear', verbose=False, enable_plotting=False)
        z_uk, ss_uk = uk.execute('grid', gridx, gridy, backend='vectorized')
        self.assertTrue(np.allclose(z_ok, z_uk))
        self.assertTrue(np.allclose(ss_ok, ss_uk))

        z_ok, ss_ok = ok.execute('grid', gridx, gridy, backend='loop')
        z_uk, ss_uk = uk.execute('grid', gridx, gridy, backend='loop')
        self.assertTrue(np.allclose(z_ok, z_uk))
        self.assertTrue(np.allclose(ss_ok, ss_uk))

    def test_ok_backends_produce_same_result(self):

        gridx = np.linspace(1067000.0, 1072000.0, 100)
        gridy = np.linspace(241500.0, 244000.0, 100)
        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                             variogram_model='linear', verbose=False, enable_plotting=False)
        z_ok_v, ss_ok_v = ok.execute('grid', gridx, gridy, backend='vectorized')
        z_ok_l, ss_ok_l = ok.execute('grid', gridx, gridy, backend='loop')
        self.assertTrue(np.allclose(z_ok_v, z_ok_l))
        self.assertTrue(np.allclose(ss_ok_v, ss_ok_l))

    def test_uk_backends_produce_same_result(self):

        gridx = np.linspace(1067000.0, 1072000.0, 100)
        gridy = np.linspace(241500.0, 244000.0, 100)
        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='linear', verbose=False, enable_plotting=False)
        z_uk_v, ss_uk_v = uk.execute('grid', gridx, gridy, backend='vectorized')
        z_uk_l, ss_uk_l = uk.execute('grid', gridx, gridy, backend='loop')
        self.assertTrue(np.allclose(z_uk_v, z_uk_l))
        self.assertTrue(np.allclose(ss_uk_v, ss_uk_l))

    def test_kriging_tools(self):

        ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
        z_write, ss_write = ok.execute('grid', self.simple_gridx, self.simple_gridy)

        kt.write_asc_grid(self.simple_gridx, self.simple_gridy, z_write,
                          filename=os.path.join(os.getcwd(), 'test_data/temp.asc'), style=1)
        z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/temp.asc'))
        self.assertTrue(np.allclose(z_write, z_read, 0.01, 0.01))
        self.assertTrue(np.allclose(self.simple_gridx, x_read))
        self.assertTrue(np.allclose(self.simple_gridy, y_read))

        z_write, ss_write = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask)
        kt.write_asc_grid(self.simple_gridx, self.simple_gridy, z_write,
                          filename=os.path.join(os.getcwd(), 'test_data/temp.asc'), style=1)
        z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/temp.asc'))
        self.assertTrue(np.ma.allclose(z_write, np.ma.masked_where(z_read == no_data, z_read),
                                       masked_equal=True, rtol=0.01, atol=0.01))
        self.assertTrue(np.allclose(self.simple_gridx, x_read))
        self.assertTrue(np.allclose(self.simple_gridy, y_read))

        ok = OrdinaryKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2])
        z_write, ss_write = ok.execute('grid', self.simple_gridx_2, self.simple_gridy)

        kt.write_asc_grid(self.simple_gridx_2, self.simple_gridy, z_write,
                          filename=os.path.join(os.getcwd(), 'test_data/temp.asc'), style=2)
        z_read, x_read, y_read, cellsize, no_data = kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/temp.asc'))
        self.assertTrue(np.allclose(z_write, z_read, 0.01, 0.01))
        self.assertTrue(np.allclose(self.simple_gridx_2, x_read))
        self.assertTrue(np.allclose(self.simple_gridy, y_read))

        os.remove(os.path.join(os.getcwd(), 'test_data/temp.asc'))

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
        self.assertEquals(z.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertEquals(ss.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertFalse(np.all(np.isnan(z)))
        self.assertTrue(np.all(np.isfinite(ss)))
        self.assertFalse(np.all(np.isnan(ss)))

        z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy, backend='loop')
        self.assertEquals(z.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertEquals(ss.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertFalse(np.all(np.isnan(z)))
        self.assertTrue(np.all(np.isfinite(ss)))
        self.assertFalse(np.all(np.isnan(ss)))

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

        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'])
        self.assertRaises(TypeError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'],
                          specified_drift=self.simple_data[:, 0])
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', drift_terms=['specified'],
                          specified_drift=[self.simple_data[:2, 0]])

        uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['specified'],
                                   specified_drift=[self.simple_data[:, 0], self.simple_data[:, 1]])
        self.assertRaises(ValueError, uk_spec.execute, 'grid', self.simple_gridx, self.simple_gridy,
                          specified_drift_arrays=[self.simple_gridx, self.simple_gridy])
        self.assertRaises(TypeError, uk_spec.execute, 'grid', self.simple_gridx, self.simple_gridy,
                          specified_drift_arrays=self.simple_gridx)
        self.assertRaises(ValueError, uk_spec.execute, 'grid', self.simple_gridx, self.simple_gridy,
                          specified_drift_arrays=[xg])
        z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy, specified_drift_arrays=[xg, yg])

        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['regional_linear'])
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

        self.assertTrue(np.allclose(z_spec, z_lin))
        self.assertTrue(np.allclose(ss_spec, ss_lin))

        uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['specified'],
                                   specified_drift=[point_log_data])
        z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                                          specified_drift_arrays=[point_log])

        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['point_log'], point_drift=well)
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

        self.assertTrue(np.allclose(z_spec, z_lin))
        self.assertTrue(np.allclose(ss_spec, ss_lin))

        uk_spec = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['specified'],
                                   specified_drift=[self.simple_data[:, 0], self.simple_data[:, 1], point_log_data])
        z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx, self.simple_gridy,
                                          specified_drift_arrays=[xg, yg, point_log])
        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['regional_linear', 'point_log'],
                                  point_drift=well)
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)

        self.assertTrue(np.allclose(z_spec, z_lin))
        self.assertTrue(np.allclose(ss_spec, ss_lin))

    def test_uk_functional_drift(self):

        well = np.array([[1.1, 1.1, -1.0]])
        func_x = lambda x, y: x
        func_y = lambda x, y: y
        func_well = lambda x, y: - well[0, 2] * np.log(np.sqrt((x - well[0, 0])**2. + (y - well[0, 1])**2.))

        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', drift_terms=['functional'])
        self.assertRaises(TypeError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='linear', drift_terms=['functional'],
                          functional_drift=func_x)

        uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['functional'],
                                   functional_drift=[func_x, func_y])
        z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['regional_linear'])
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
        self.assertTrue(np.allclose(z_func, z_lin))
        self.assertTrue(np.allclose(ss_func, ss_lin))

        uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['functional'], functional_drift=[func_well])
        z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['point_log'], point_drift=well)
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
        self.assertTrue(np.allclose(z_func, z_lin))
        self.assertTrue(np.allclose(ss_func, ss_lin))

        uk_func = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                   variogram_model='linear', drift_terms=['functional'],
                                   functional_drift=[func_x, func_y, func_well])
        z_func, ss_func = uk_func.execute('grid', self.simple_gridx, self.simple_gridy)
        uk_lin = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                                  variogram_model='linear', drift_terms=['regional_linear', 'point_log'],
                                  point_drift=well)
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx, self.simple_gridy)
        self.assertTrue(np.allclose(z_func, z_lin))
        self.assertTrue(np.allclose(ss_func, ss_lin))

    def test_uk_with_external_drift(self):

        dem, demx, demy, cellsize, no_data = \
            kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/test3_dem.asc'))
        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='spherical',
                              variogram_parameters=[500.0, 3000.0, 0.0],
                              anisotropy_scaling=1.0, anisotropy_angle=0.0,
                              drift_terms=['external_Z'], external_drift=dem,
                              external_drift_x=demx, external_drift_y=demy,
                              verbose=False)
        answer, gridx, gridy, cellsize, no_data = \
            kt.read_asc_grid(os.path.join(os.getcwd(), 'test_data/test3_answer.asc'))

        z, ss = uk.execute('grid', gridx, gridy, backend='vectorized')
        self.assertTrue(np.allclose(z, answer))

        z, ss = uk.execute('grid', gridx, gridy, backend='loop')
        self.assertTrue(np.allclose(z, answer))

    def test_force_exact(self):
        data = np.array([[1., 1., 2.],
                         [2., 2., 1.5],
                         [3., 3., 1.]])
        ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                             variogram_model='linear', variogram_parameters=[1.0, 1.0])
        z, ss = ok.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='vectorized')
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = ok.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.], backend='vectorized')
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = ok.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1), backend='vectorized')
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
        z, ss = ok.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend='vectorized')
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = ok.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend='vectorized')
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = ok.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25), backend='vectorized',
                           mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))

        z, ss = ok.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='loop')
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = ok.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.], backend='loop')
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = ok.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1), backend='loop')
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
        z, ss = ok.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend='loop')
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = ok.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend='loop')
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = ok.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25), backend='loop',
                           mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))

        uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2])
        z, ss = uk.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='vectorized')
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = uk.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.], backend='vectorized')
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = uk.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1), backend='vectorized')
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
        z, ss = uk.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend='vectorized')
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = uk.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend='vectorized')
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = uk.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25), backend='vectorized',
                           mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))
        z, ss = uk.execute('grid', [1., 2., 3.], [1., 2., 3.], backend='loop')
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = uk.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.], backend='loop')
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = uk.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1), backend='loop')
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
        z, ss = uk.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1), backend='loop')
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = uk.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1), backend='loop')
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = uk.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25), backend='loop',
                           mask=np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))

        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (1., 1.),
                           variogram_models.linear_variogram_model, [1.0, 1.0],
                           'euclidean')
        self.assertAlmostEqual(z, 2.)
        self.assertAlmostEqual(ss, 0.)
        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (1., 2.),
                           variogram_models.linear_variogram_model, [1.0, 1.0],
                           'euclidean')
        self.assertNotAlmostEqual(ss, 0.)

        data = np.zeros((50, 3))
        x, y = np.meshgrid(np.arange(0., 10., 1.), np.arange(0., 10., 2.))
        data[:, 0] = np.ravel(x)
        data[:, 1] = np.ravel(y)
        data[:, 2] = np.ravel(x) * np.ravel(y)
        ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                             variogram_model='linear', variogram_parameters=[100.0, 1.0])
        z, ss = ok.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.), backend='vectorized')
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = ok.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.), backend='vectorized')
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))
        z, ss = ok.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.), backend='loop')
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = ok.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.), backend='loop')
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))

        uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                              variogram_model='linear', variogram_parameters=[100.0, 1.0])
        z, ss = uk.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.), backend='vectorized')
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = uk.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.), backend='vectorized')
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))
        z, ss = uk.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.), backend='loop')
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = uk.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.), backend='loop')
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))

    def test_custom_variogram(self):
        func = lambda params, dist: params[0] * np.log10(dist + params[1]) + params[2]

        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='mrow')
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='custom')
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='custom', variogram_function=0)
        self.assertRaises(ValueError, UniversalKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='custom', variogram_function=func)
        uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
        self.assertAlmostEqual(uk.variogram_function([1., 1., 1.], 1.), 1.3010, 4)
        uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear')
        uk.update_variogram_model('custom', variogram_parameters=[1., 1., 1.], variogram_function=func)
        self.assertAlmostEqual(uk.variogram_function([1., 1., 1.], 1.), 1.3010, 4)

        self.assertRaises(ValueError, OrdinaryKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='mrow')
        self.assertRaises(ValueError, OrdinaryKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='custom')
        self.assertRaises(ValueError, OrdinaryKriging, self.simple_data[:, 0], self.simple_data[:, 1],
                          self.simple_data[:, 2], variogram_model='custom', variogram_function=0)
        self.assertRaises(ValueError, OrdinaryKriging, self.simple_data[:, 0], self.simple_data[:, 1],
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
        self.assertTrue(np.allclose(k, self.ok_test_answer))
        k, ss = k3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='loop')
        self.assertTrue(np.allclose(k, self.ok_test_answer))

        # Test to compare K3D results to those obtained using KT3D.
        data = np.genfromtxt('./test_data/test3d_data.txt', skip_header=1)
        ans = np.genfromtxt('./test_data/test3d_answer.txt')
        ans_z = ans[:, 0].reshape((10, 10, 10))
        ans_ss = ans[:, 1].reshape((10, 10, 10))
        k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                variogram_model='linear', variogram_parameters=[1., 0.1])
        k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
        self.assertTrue(np.allclose(k, ans_z, rtol=1e-3))
        self.assertTrue(np.allclose(ss, ans_ss, rtol=1e-3))
        k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                variogram_model='linear', variogram_parameters=[1., 0.1])
        k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')
        self.assertTrue(np.allclose(k, ans_z, rtol=1e-3))
        self.assertTrue(np.allclose(ss, ans_ss, rtol=1e-3))

    def test_ok3d_moving_window(self):

        # Test to compare K3D results to those obtained using KT3D.
        data = np.genfromtxt('./test_data/test3d_data.txt', skip_header=1)
        ans = np.genfromtxt('./test_data/test3d_answer.txt')
        ans_z = ans[:, 0].reshape((10, 10, 10))
        ans_ss = ans[:, 1].reshape((10, 10, 10))
        k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                variogram_model='linear', variogram_parameters=[1., 0.1])
        k, ss = k3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop', n_closest_points=10)
        self.assertTrue(np.allclose(k, ans_z, rtol=1e-3))
        self.assertTrue(np.allclose(ss, ans_ss, rtol=1e-3))

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
        self.assertTrue(np.allclose(uk_v, ok_v))
        uk_l, uss_l = uk3d.execute('grid', self.ok_test_gridx, self.ok_test_gridy, np.array([0.]), backend='loop')
        self.assertTrue(np.allclose(uk_l, ok_l))
        self.assertTrue(np.allclose(uk_l, uk_v))
        self.assertTrue(np.allclose(uss_l, uss_v))

        data = np.genfromtxt('./test_data/test3d_data.txt', skip_header=1)
        ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                 variogram_model='linear', variogram_parameters=[1., 0.1])
        ok_v, oss_v = ok3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
        ok_l, oss_l = ok3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')

        uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                  variogram_model='linear', variogram_parameters=[1., 0.1])
        uk_v, uss_v = uk3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='vectorized')
        self.assertTrue(np.allclose(uk_v, ok_v))
        self.assertTrue(np.allclose(uss_v, oss_v))
        uk_l, uss_l = uk3d.execute('grid', np.arange(10.), np.arange(10.), np.arange(10.), backend='loop')
        self.assertTrue(np.allclose(uk_l, ok_l))
        self.assertTrue(np.allclose(uss_l, oss_l))
        self.assertTrue(np.allclose(uk_l, uk_v))
        self.assertTrue(np.allclose(uss_l, uss_v))

    def test_ok3d_update_variogram_model(self):

        self.assertRaises(ValueError, OrdinaryKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
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

        self.assertRaises(ValueError, k3d.update_variogram_model, 'blurg')
        k3d.update_variogram_model('power', anisotropy_scaling_y=3.0, anisotropy_scaling_z=3.0,
                                   anisotropy_angle_x=45.0, anisotropy_angle_y=45.0, anisotropy_angle_z=45.0)
        self.assertFalse(variogram_model == k3d.variogram_model)
        self.assertFalse(variogram_parameters == k3d.variogram_model_parameters)
        self.assertFalse(anisotropy_scaling_y == k3d.anisotropy_scaling_y)
        self.assertFalse(anisotropy_scaling_z == k3d.anisotropy_scaling_z)
        self.assertFalse(anisotropy_angle_x == k3d.anisotropy_angle_x)
        self.assertFalse(anisotropy_angle_y == k3d.anisotropy_angle_y)
        self.assertFalse(anisotropy_angle_z == k3d.anisotropy_angle_z)

    def test_uk3d_update_variogram_model(self):

        self.assertRaises(ValueError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
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

        self.assertRaises(ValueError, uk3d.update_variogram_model, 'blurg')
        uk3d.update_variogram_model('power', anisotropy_scaling_y=3.0, anisotropy_scaling_z=3.0,
                                    anisotropy_angle_x=45.0, anisotropy_angle_y=45.0, anisotropy_angle_z=45.0)
        self.assertFalse(variogram_model == uk3d.variogram_model)
        self.assertFalse(variogram_parameters == uk3d.variogram_model_parameters)
        self.assertFalse(anisotropy_scaling_y == uk3d.anisotropy_scaling_y)
        self.assertFalse(anisotropy_scaling_z == uk3d.anisotropy_scaling_z)
        self.assertFalse(anisotropy_angle_x == uk3d.anisotropy_angle_x)
        self.assertFalse(anisotropy_angle_y == uk3d.anisotropy_angle_y)
        self.assertFalse(anisotropy_angle_z == uk3d.anisotropy_angle_z)

    def test_ok3d_backends_produce_same_result(self):

        k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                self.simple_data_3d[:, 3], variogram_model='linear')
        k_k3d_v, ss_k3d_v = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                        backend='vectorized')
        k_k3d_l, ss_k3d_l = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                        backend='loop')
        self.assertTrue(np.allclose(k_k3d_v, k_k3d_l))
        self.assertTrue(np.allclose(ss_k3d_v, ss_k3d_l))

    def test_ok3d_execute(self):

        k3d = OrdinaryKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                                self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])

        self.assertRaises(ValueError, k3d.execute, 'blurg', self.simple_gridx_3d,
                          self.simple_gridy_3d, self.simple_gridz_3d)

        k, ss = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                            self.simple_gridz_3d, backend='vectorized')
        shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
        self.assertEqual(k.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(k), np.amin(k))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(k))

        k, ss = k3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                            self.simple_gridz_3d, backend='loop')
        shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
        self.assertEqual(k.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(k), np.amin(k))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(k))

        self.assertRaises(IOError, k3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, backend='vectorized')
        mask = np.array([True, False])
        self.assertRaises(ValueError, k3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, mask=mask, backend='vectorized')
        k, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                            mask=self.mask_3d, backend='vectorized')
        self.assertTrue(np.ma.is_masked(k))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(k[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)
        z, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                            mask=self.mask_3d.T, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)

        self.assertRaises(IOError, k3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, backend='loop')
        mask = np.array([True, False])
        self.assertRaises(ValueError, k3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, mask=mask, backend='loop')
        k, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                            mask=self.mask_3d, backend='loop')
        self.assertTrue(np.ma.is_masked(k))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(k[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)
        z, ss = k3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                            mask=self.mask_3d.T, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)

        self.assertRaises(ValueError, k3d.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          np.array([1.0]), backend='vectorized')
        k, ss = k3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                            self.simple_gridz_3d[0], backend='vectorized')
        self.assertEqual(k.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        self.assertRaises(ValueError, k3d.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          np.array([1.0]), backend='loop')
        k, ss = k3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                            self.simple_gridz_3d[0], backend='loop')
        self.assertEqual(k.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        data = np.zeros((125, 4))
        z, y, x = np.meshgrid(np.arange(0., 5., 1.), np.arange(0., 5., 1.), np.arange(0., 5., 1.))
        data[:, 0] = np.ravel(x)
        data[:, 1] = np.ravel(y)
        data[:, 2] = np.ravel(z)
        data[:, 3] = np.ravel(z)
        k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
        k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                            np.arange(0., 4., 1.), backend='vectorized')
        self.assertTrue(np.allclose(k[0, :, :], 0., atol=0.01))
        self.assertTrue(np.allclose(k[1, :, :], 1., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2, :, :], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[3, :, :], 3., rtol=1.e-2))
        k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                            np.arange(0., 4., 1.), backend='loop')
        self.assertTrue(np.allclose(k[0, :, :], 0., atol=0.01))
        self.assertTrue(np.allclose(k[1, :, :], 1., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2, :, :], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[3, :, :], 3., rtol=1.e-2))
        k3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
        k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='vectorized')
        self.assertTrue(np.allclose(k[0], 1., atol=0.01))
        self.assertTrue(np.allclose(k[1], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2], 3., rtol=1.e-2))
        k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='loop')
        self.assertTrue(np.allclose(k[0], 1., atol=0.01))
        self.assertTrue(np.allclose(k[1], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2], 3., rtol=1.e-2))

    def test_uk3d_execute(self):

        uk3d = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                                  self.simple_data_3d[:, 2], self.simple_data_3d[:, 3])

        self.assertRaises(ValueError, uk3d.execute, 'blurg', self.simple_gridx_3d,
                          self.simple_gridy_3d, self.simple_gridz_3d)

        k, ss = uk3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                             self.simple_gridz_3d, backend='vectorized')
        shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
        self.assertEqual(k.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(k), np.amin(k))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(k))

        k, ss = uk3d.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d,
                             self.simple_gridz_3d, backend='loop')
        shape = (self.simple_gridz_3d.size, self.simple_gridy_3d.size, self.simple_gridx_3d.size)
        self.assertEqual(k.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(k), np.amin(k))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(k))

        self.assertRaises(IOError, uk3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, backend='vectorized')
        mask = np.array([True, False])
        self.assertRaises(ValueError, uk3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, mask=mask, backend='vectorized')
        k, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                             mask=self.mask_3d, backend='vectorized')
        self.assertTrue(np.ma.is_masked(k))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(k[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)
        z, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                             mask=self.mask_3d.T, backend='vectorized')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)

        self.assertRaises(IOError, uk3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, backend='loop')
        mask = np.array([True, False])
        self.assertRaises(ValueError, uk3d.execute, 'masked', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, mask=mask, backend='loop')
        k, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                             mask=self.mask_3d, backend='loop')
        self.assertTrue(np.ma.is_masked(k))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(k[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)
        z, ss = uk3d.execute('masked', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                             mask=self.mask_3d.T, backend='loop')
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0, 0], np.ma.masked)
        self.assertIs(ss[0, 0, 0], np.ma.masked)

        self.assertRaises(ValueError, uk3d.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          np.array([1.0]), backend='vectorized')
        k, ss = uk3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                             self.simple_gridz_3d[0], backend='vectorized')
        self.assertEqual(k.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        self.assertRaises(ValueError, uk3d.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                          np.array([1.0]), backend='loop')
        k, ss = uk3d.execute('points', self.simple_gridx_3d[0], self.simple_gridy_3d[0],
                             self.simple_gridz_3d[0], backend='loop')
        self.assertEqual(k.shape, (1,))
        self.assertEqual(ss.shape, (1,))

        data = np.zeros((125, 4))
        z, y, x = np.meshgrid(np.arange(0., 5., 1.), np.arange(0., 5., 1.), np.arange(0., 5., 1.))
        data[:, 0] = np.ravel(x)
        data[:, 1] = np.ravel(y)
        data[:, 2] = np.ravel(z)
        data[:, 3] = np.ravel(z)
        k3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
        k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                            np.arange(0., 4., 1.), backend='vectorized')
        self.assertTrue(np.allclose(k[0, :, :], 0., atol=0.01))
        self.assertTrue(np.allclose(k[1, :, :], 1., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2, :, :], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[3, :, :], 3., rtol=1.e-2))
        k, ss = k3d.execute('grid', np.arange(2., 3., 0.1), np.arange(2., 3., 0.1),
                            np.arange(0., 4., 1.), backend='loop')
        self.assertTrue(np.allclose(k[0, :, :], 0., atol=0.01))
        self.assertTrue(np.allclose(k[1, :, :], 1., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2, :, :], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[3, :, :], 3., rtol=1.e-2))
        k3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3], variogram_model='linear')
        k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='vectorized')
        self.assertTrue(np.allclose(k[0], 1., atol=0.01))
        self.assertTrue(np.allclose(k[1], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2], 3., rtol=1.e-2))
        k, ss = k3d.execute('points', [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [1., 2., 3.], backend='loop')
        self.assertTrue(np.allclose(k[0], 1., atol=0.01))
        self.assertTrue(np.allclose(k[1], 2., rtol=1.e-2))
        self.assertTrue(np.allclose(k[2], 3., rtol=1.e-2))

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

        self.assertRaises(ValueError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                          self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                          variogram_model='linear', drift_terms=['specified'])
        self.assertRaises(TypeError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                          self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='linear',
                          drift_terms=['specified'], specified_drift=self.simple_data_3d[:, 0])
        self.assertRaises(ValueError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                          self.simple_data_3d[:, 2], self.simple_data_3d[:, 3], variogram_model='linear',
                          drift_terms=['specified'], specified_drift=[self.simple_data_3d[:2, 0]])

        uk_spec = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                     self.simple_data_3d[:, 3], variogram_model='linear', drift_terms=['specified'],
                                     specified_drift=[self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                                                      self.simple_data_3d[:, 2]])
        self.assertRaises(ValueError, uk_spec.execute, 'grid', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, specified_drift_arrays=[self.simple_gridx_3d, self.simple_gridy_3d,
                                                                        self.simple_gridz_3d])
        self.assertRaises(TypeError, uk_spec.execute, 'grid', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, specified_drift_arrays=self.simple_gridx_3d)
        self.assertRaises(ValueError, uk_spec.execute, 'grid', self.simple_gridx_3d, self.simple_gridy_3d,
                          self.simple_gridz_3d, specified_drift_arrays=[zg])
        z_spec, ss_spec = uk_spec.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d,
                                          specified_drift_arrays=[xg, yg, zg])

        uk_lin = UniversalKriging3D(self.simple_data_3d[:, 0], self.simple_data_3d[:, 1], self.simple_data_3d[:, 2],
                                    self.simple_data_3d[:, 3], variogram_model='linear',
                                    drift_terms=['regional_linear'])
        z_lin, ss_lin = uk_lin.execute('grid', self.simple_gridx_3d, self.simple_gridy_3d, self.simple_gridz_3d)

        self.assertTrue(np.allclose(z_spec, z_lin))
        self.assertTrue(np.allclose(ss_spec, ss_lin))

    def test_uk3d_functional_drift(self):

        func_x = lambda x, y, z: x
        func_y = lambda x, y, z: y
        func_z = lambda x, y, z: z

        self.assertRaises(ValueError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
                          self.simple_data_3d[:, 2], self.simple_data_3d[:, 3],
                          variogram_model='linear', drift_terms=['functional'])
        self.assertRaises(TypeError, UniversalKriging3D, self.simple_data_3d[:, 0], self.simple_data_3d[:, 1],
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
        self.assertTrue(np.allclose(z_func, z_lin))
        self.assertTrue(np.allclose(ss_func, ss_lin))

    def test_geometric_code(self):
        
        # Create selected points distributed across the sphere:
        N=4
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


@unittest.skipUnless(SKLEARN_INSTALLED, "scikit-learn not installed")
class TestKrige(unittest.TestCase):

    @staticmethod
    def method_and_vergiogram():
        method = ['ordinary', 'universal', 'ordinary3d', 'universal3d']
        variogram_model = ['linear', 'power', 'gaussian', 'spherical',
                           'exponential']
        return product(method, variogram_model)

    def test_krige(self):
        from pykrige.rk import Krige
        from pykrige.rk import threed_krige
        from pykrige.compat import GridSearchCV
        # dummy data
        np.random.seed(1)
        X = np.random.randint(0, 400, size=(20, 3)).astype(float)
        y = 5 * np.random.rand(20)

        for m, v in self.method_and_vergiogram():
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


@unittest.skipUnless(SKLEARN_INSTALLED, "scikit-learn not installed")
class TestRegressionKrige(unittest.TestCase):

    @staticmethod
    def methods():
        from sklearn.svm import SVR
        from sklearn.linear_model import ElasticNet, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        krige_methods = ['ordinary', 'universal']
        ml_methods = [SVR(C=0.01),
                      RandomForestRegressor(min_samples_split=5,
                                            n_estimators=50),
                      LinearRegression(),
                      Lasso(),
                      ElasticNet()
                      ]
        return product(ml_methods, krige_methods)

    def test_krige(self):
        from pykrige.rk import RegressionKriging
        from pykrige.compat import train_test_split
        from itertools import product
        np.random.seed(1)
        x = np.linspace(-1., 1., 100)
        # create a feature matrix with 5 features
        X = np.tile(x, reps=(5, 1)).T
        y = 1 + 5*X[:, 0] - 2*X[:, 1] - 2*X[:, 2] + 3*X[:, 3] + 4*X[:, 4] + \
            2*(np.random.rand(100) - 0.5)

        # create lat/lon array
        lon = np.linspace(-180., 180.0, 10)
        lat = np.linspace(-90., 90., 10)
        lon_lat = np.array(list(product(lon, lat)))

        X_train, X_test, y_train, y_test, lon_lat_train, lon_lat_test = \
            train_test_split(X, y, lon_lat, train_size=0.7, random_state=10)

        for ml_model, krige_method in self.methods():
            reg_kr_model = RegressionKriging(regression_model=ml_model,
                                             method=krige_method,
                                             n_closest_points=2)
            reg_kr_model.fit(X_train, lon_lat_train, y_train)
            assert reg_kr_model.score(X_test, lon_lat_test, y_test) > 0.25

    def test_krige_housing(self):
        from pykrige.rk import RegressionKriging
        from pykrige.compat import train_test_split
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()

        # take only first 1000
        p = housing['data'][:1000, :-2]
        x = housing['data'][:1000, -2:]
        target = housing['target'][:1000]

        p_train, p_test, y_train, y_test, x_train, x_test = \
            train_test_split(p, target, x, train_size=0.7,
                             random_state=10)

        for ml_model, krige_method in self.methods():

            reg_kr_model = RegressionKriging(regression_model=ml_model,
                                             method=krige_method,
                                             n_closest_points=2)
            reg_kr_model.fit(p_train, x_train, y_train)
            if krige_method == 'ordinary':
                assert reg_kr_model.score(p_test, x_test, y_test) > 0.5
            else:
                assert reg_kr_model.score(p_test, x_test, y_test) > 0.0


if __name__ == '__main__':
    unittest.main()
