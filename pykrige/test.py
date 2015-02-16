"""
Testing code.
BSM Jan 2015
"""

import unittest
import os
import numpy as np
import kriging_tools as kt
import core
import variogram_models
from ok import OrdinaryKriging
from uk import UniversalKriging


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

    def test_core_adjust_for_anisotropy(self):

        x = np.array([1.0, 0.0, -1.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        rotated_x, rotated_y = core.adjust_for_anisotropy(x, y, 0.0, 0.0, 2.0, 90.0)
        self.assertTrue(np.allclose(rotated_x, np.array([0.0, -1.0, 0.0, 1.0])))
        self.assertTrue(np.allclose(rotated_y, np.array([2.0, 0.0, -2.0, 0.0])))

    def test_core_initialize_variogram_model(self):

        # Note the variogram_function argument is not a string in real life...
        self.assertRaises(ValueError, core.initialize_variogram_model, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], 'linear', [0.0], 'linear', 6, False)

        self.assertRaises(ValueError, core.initialize_variogram_model, self.test_data[:, 0], self.test_data[:, 1],
                          self.test_data[:, 2], 'spherical', [0.0], 'spherical', 6, False)

        x = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
        y = np.array([1.0 + n/np.sqrt(2) for n in range(4)])
        z = np.arange(1.0, 5.0, 1.0)
        lags, semivariance, variogram_model_parameters = core.initialize_variogram_model(x, y, z, 'linear',
                                                                                         [0.0, 0.0], 'linear',
                                                                                         6, False)

        self.assertTrue(np.allclose(lags, np.array([1.0, 2.0, 3.0])))
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
                           variogram_models.linear_variogram_model, [0.006, 0.1])
        self.assertAlmostEqual(z, 1.6364, 4)
        self.assertAlmostEqual(ss, 0.4201, 4)

        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (43.8, 24.6),
                           variogram_models.linear_variogram_model, [0.006, 0.1])
        self.assertAlmostEqual(z, 2.822, 3)
        self.assertAlmostEqual(ss, 0.0, 3)

    def test_ok(self):

        # Test to compare OK results to those obtained using KT3D_H2O.
        # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                             variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0])
        z, ss = ok.execute('grid', self.ok_test_gridx, self.ok_test_gridy)
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

        z, ss = ok.execute('grid', self.simple_gridx, self.simple_gridy)
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        self.assertRaises(IOError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy)
        mask = np.array([True, False])
        self.assertRaises(ValueError, ok.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask)
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask)
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = ok.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T)
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(ValueError, ok.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
        z, ss = ok.execute('points', self.simple_gridx[0], self.simple_gridy[0])
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

    def test_uk(self):

        # Test to compare UK with linear drift to results from KT3D_H2O.
        # (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)

        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='exponential', variogram_parameters=[500.0, 3000.0, 0.0],
                              drift_terms=['regional_linear'])
        z, ss = uk.execute('grid', self.uk_test_gridx, self.uk_test_gridy)
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
        z, ss = uk.execute('points', np.array([point[0]]), np.array([point[1]]))
        self.assertAlmostEqual(z_answer, z, places=0)
        self.assertAlmostEqual(ss_answer, ss, places=0)

        z, ss = uk.execute('points', np.array([61.0]), np.array([139.0]))
        self.assertAlmostEqual(z, 477.0, 3)
        self.assertAlmostEqual(ss, 0.0, 3)

    def test_uk_execute(self):

        uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear'])

        self.assertRaises(ValueError, uk.execute, 'blurg', self.simple_gridx, self.simple_gridy)

        z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy)
        shape = (self.simple_gridy.size, self.simple_gridx.size)
        self.assertEqual(z.shape, shape)
        self.assertEqual(ss.shape, shape)
        self.assertNotEqual(np.amax(z), np.amin(z))
        self.assertNotEqual(np.amax(ss), np.amin(ss))
        self.assertFalse(np.ma.is_masked(z))

        self.assertRaises(IOError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy)
        mask = np.array([True, False])
        self.assertRaises(ValueError, uk.execute, 'masked', self.simple_gridx, self.simple_gridy, mask=mask)
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask)
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)
        z, ss = uk.execute('masked', self.simple_gridx, self.simple_gridy, mask=self.mask.T)
        self.assertTrue(np.ma.is_masked(z))
        self.assertTrue(np.ma.is_masked(ss))
        self.assertIs(z[0, 0], np.ma.masked)
        self.assertIs(ss[0, 0], np.ma.masked)

        self.assertRaises(ValueError, uk.execute, 'points', np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
        z, ss = uk.execute('points', self.simple_gridx[0], self.simple_gridy[0])
        self.assertEqual(z.shape, (1,))
        self.assertEqual(ss.shape, (1,))

    def test_ok_uk_produce_same_result(self):

        gridx = np.linspace(1067000.0, 1072000.0, 100)
        gridy = np.linspace(241500.0, 244000.0, 100)
        ok = OrdinaryKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                             variogram_model='linear', verbose=False, enable_plotting=False)
        z_ok, ss_ok = ok.execute('grid', gridx, gridy)
        uk = UniversalKriging(self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2],
                              variogram_model='linear', verbose=False, enable_plotting=False)
        z_uk, ss_uk = uk.execute('grid', gridx, gridy)

        self.assertTrue(np.allclose(z_ok, z_uk))
        self.assertTrue(np.allclose(ss_ok, ss_uk))

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

    def test_uk_everything(self):

        well = np.array([[1.1, 1.1, -1.0]])
        dem = np.arange(0.0, 5.1, 0.1)
        dem = np.repeat(dem[np.newaxis, :], 6, axis=0)
        dem_x = np.arange(0.0, 5.1, 0.1)
        dem_y = np.arange(0.0, 6.0, 1.0)

        uk = UniversalKriging(self.simple_data[:, 0], self.simple_data[:, 1], self.simple_data[:, 2],
                              variogram_model='linear', drift_terms=['regional_linear', 'external_Z', 'point_log'],
                              point_drift=well, external_drift=dem, external_drift_x=dem_x, external_drift_y=dem_y)
        z, ss = uk.execute('grid', self.simple_gridx, self.simple_gridy)

        self.assertEquals(z.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertEquals(ss.shape, (self.simple_gridy.shape[0], self.simple_gridx.shape[0]))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertFalse(np.all(np.isnan(z)))
        self.assertTrue(np.all(np.isfinite(ss)))
        self.assertFalse(np.all(np.isnan(ss)))

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
        z, ss = uk.execute('grid', gridx, gridy)

        self.assertTrue(np.allclose(z, answer))

    def test_force_exact(self):
        data = np.array([[1., 1., 2.],
                         [2., 2., 1.5],
                         [3., 3., 1.]])
        ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                             variogram_model='linear', variogram_parameters=[1.0, 1.0])
        z, ss = ok.execute('grid', [1., 2., 3.], [1., 2., 3.])
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = ok.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.])
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = ok.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1))
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
        z, ss = ok.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1))
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = ok.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1))
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = ok.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25),
                           np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))

        uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2])
        z, ss = uk.execute('grid', [1., 2., 3.], [1., 2., 3.])
        self.assertAlmostEqual(z[0, 0], 2.0)
        self.assertAlmostEqual(ss[0, 0], 0.0)
        self.assertAlmostEqual(z[1, 1], 1.5)
        self.assertAlmostEqual(ss[1, 1], 0.0)
        self.assertAlmostEqual(z[2, 2], 1.0)
        self.assertAlmostEqual(ss[2, 2], 0.0)
        self.assertNotAlmostEqual(ss[0, 2], 0.0)
        self.assertNotAlmostEqual(ss[2, 0], 0.0)
        z, ss = uk.execute('points', [1., 2., 3., 3.], [2., 1., 1., 3.])
        self.assertNotAlmostEqual(ss[0], 0.0)
        self.assertNotAlmostEqual(ss[1], 0.0)
        self.assertNotAlmostEqual(ss[2], 0.0)
        self.assertAlmostEqual(z[3], 1.0)
        self.assertAlmostEqual(ss[3], 0.0)
        z, ss = uk.execute('grid', np.arange(0., 4., 0.1), np.arange(0., 4., 0.1))
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
        z, ss = uk.execute('grid', np.arange(0., 3.1, 0.1), np.arange(2.1, 3.1, 0.1))
        self.assertTrue(np.any(ss <= 1e-15))
        self.assertFalse(np.any(ss[:9, :30] <= 1e-15))
        self.assertFalse(np.allclose(z[:9, :30], 0.))
        z, ss = uk.execute('grid', np.arange(0., 1.9, 0.1), np.arange(2.1, 3.1, 0.1))
        self.assertFalse(np.any(ss <= 1e-15))
        z, ss = uk.execute('masked', np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25),
                           np.asarray(np.meshgrid(np.arange(2.5, 3.5, 0.1), np.arange(2.5, 3.5, 0.25))[0] == 0.))
        self.assertTrue(ss[2, 5] <= 1e-15)
        self.assertFalse(np.allclose(ss, 0.))

        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (1., 1.),
                           variogram_models.linear_variogram_model, [1.0, 1.0])
        self.assertAlmostEqual(z, 2.)
        self.assertAlmostEqual(ss, 0.)
        z, ss = core.krige(data[:, 0], data[:, 1], data[:, 2], (1., 2.),
                           variogram_models.linear_variogram_model, [1.0, 1.0])
        self.assertNotAlmostEqual(ss, 0.)

        data = np.zeros((50, 3))
        x, y = np.meshgrid(np.arange(0., 10., 1.), np.arange(0., 10., 2.))
        data[:, 0] = np.ravel(x)
        data[:, 1] = np.ravel(y)
        data[:, 2] = np.ravel(x) * np.ravel(y)
        ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                             variogram_model='linear', variogram_parameters=[100.0, 1.0])
        z, ss = ok.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.))
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = ok.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.))
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))

        uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                              variogram_model='linear', variogram_parameters=[100.0, 1.0])
        z, ss = uk.execute('grid', np.arange(0., 10., 1.), np.arange(0., 10., 2.))
        self.assertTrue(np.allclose(np.ravel(z), data[:, 2]))
        self.assertTrue(np.allclose(ss, 0.))
        z, ss = uk.execute('grid', np.arange(0.5, 10., 1.), np.arange(0.5, 10., 2.))
        self.assertFalse(np.allclose(np.ravel(z), data[:, 2]))
        self.assertFalse(np.allclose(ss, 0.))


if __name__ == '__main__':
    unittest.main()