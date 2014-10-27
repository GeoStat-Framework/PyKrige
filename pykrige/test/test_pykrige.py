from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import pykrige.kriging_tools as kt
import numpy as np


def test_uk_ok_produce_same_result(data):

    print "Testing OK/UK equivalence..."

    gridx = np.linspace(1067000.0, 1072000.0, 100)
    gridy = np.linspace(241500.0, 244000.0, 100)
    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                         verbose=False, enable_plotting=False)
    z_ok, ss_ok = ok.execute(gridx, gridy)
    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                          verbose=False, enable_plotting=False)
    z_uk, ss_uk = uk.execute(gridx, gridy)

    if np.array_equal(z_ok, z_uk):
        z = True
    else:
        z = False
    if np.array_equal(ss_ok, ss_uk):
        ss = True
    else:
        ss = False
    if z:
        print "Kriged grid from UK matches that from OK."
    else:
        print "Kriged grids DO NOT match!"
    if ss:
        print "SS from UK matches that from OK."
    else:
        print "Errors DO NOT match!"
    print '\n'


def test_linear_drift():
    """
    Test to ensure that UK linear drift behaves as expected...
    Data and answer from lecture notes by Nicolas Christou, UCLA Stats
    """

    print "Testing UK with linear drifts..."

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

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                          variogram_model='exponential',
                          variogram_parameters=[10.0, 9.99, 0.0],
                          drift_terms=['regional_linear'])
    z, ss = uk.execute(point[0], point[1])

    z_diff = 100.0 * abs(z_answer - z)/z_answer
    ss_diff = 100.0 * abs(ss_answer - ss)/ss_answer
    
    print "Calculated Z differs from expected Z by %.4f%%" % z_diff
    print "Calculated SigmaSq differs from expected SigmaSq by %.4f%%" % ss_diff
    if z_diff <= 0.1 and ss_diff <= 0.1:
        print "Acceptable."
    else:
        print "Difference unacceptable!"
    print '\n'


def test1(data):
    """
    Test to compare OK results to those obtained using KT3D_H2O.
    (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)
    """

    print "Comparing OK to KT3D_H2O..."

    ok = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2],
                         variogram_model='exponential',
                         variogram_parameters=[500.0, 3000.0, 0.0],
                         anisotropy_scaling=1.0, anisotropy_angle=0.0)
    answer, gridx, gridy, cellsize, no_data = kt.read_asc_grid("test1_answer.asc", footer=2)
    z, ss = ok.execute(gridx, gridy)
    diff = 100.0*np.absolute(z - answer)/answer
    mean_diff = np.mean(diff)
    max_diff = np.amax(diff)
    print "Mean percent difference is %.4f%%" % mean_diff
    print "Max percent difference is %.4f%%" % max_diff
    if mean_diff <= 0.1 and max_diff <= 0.1:
        print "Acceptable."
    else:
        print "Difference unacceptable!"
    print '\n'


def test2(data):
    """
    Test to compare UK with linear drift to results from KT3D_H2O.
    (M. Karanovic, M. Tonkin, and D. Wilson, 2009, Groundwater, vol. 47, no. 4, 580-586.)
    """

    print "Comparing UK with linear drift to KT3D_H2O..."

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                          variogram_model='exponential',
                          variogram_parameters=[500.0, 3000.0, 0.0],
                          anisotropy_scaling=1.0, anisotropy_angle=0.0,
                          drift_terms=['regional_linear'])
    answer, gridx, gridy, cellsize, no_data = kt.read_asc_grid("test2_answer.asc", footer=2)
    z, ss = uk.execute(gridx, gridy)
    diff = 100.0 * np.absolute(z - answer)/answer
    mean_diff = np.mean(diff)
    max_diff = np.amax(diff)
    print "Mean percent difference is %.4f%%" % mean_diff
    print "Max percent difference is %.4f%%" % max_diff
    if mean_diff <= 0.1 and max_diff <= 0.1:
        print "Acceptable."
    else:
        print "Difference unacceptable!"
    print '\n'


def test3(data):
    """
    Test to compare UK with DEM drift to results of MEUK.
    (MEUK code by J. Kennel and M. Tonkin, S.S. Papadopulos & Assoc., Inc.;
    "A Hybrid Analytic Element Universal Kriging Interpolation Technique
    Built in the Open Source R Environment", presented at AGU 2013 Fall Meeting;
    http://adsabs.harvard.edu/abs/2013AGUFM.H52E..03T)

    !!! THIS TEST IS UNUSABLE !!!
    """

    print "Comparing UK with DEM drift to MEUK..."

    dem, demx, demy, cellsize, no_data = kt.read_asc_grid("test3_dem.asc")

    uk = UniversalKriging(data[:, 0], data[:, 1], data[:, 2],
                          variogram_model='spherical',
                          variogram_parameters=[500.0, 3000.0, 0.0],
                          anisotropy_scaling=1.0, anisotropy_angle=0.0,
                          drift_terms=['external_Z'], external_drift=dem,
                          external_drift_x=demx, external_drift_y=demy,
                          verbose=False)
    uk.UNBIAS = True
    answer, gridx, gridy, cellsize, no_data = kt.read_asc_grid("test3_answer.asc")
    z, ss = uk.execute(gridx, gridy)

    abs_diff = np.absolute(z - answer)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.amax(abs_diff)
    perc_diff = 100.0*np.absolute(z - answer)/answer
    mean_perc_diff = np.mean(perc_diff)
    max_perc_diff = np.amax(perc_diff)
    print "Mean absolute difference is %.4f" % mean_abs_diff
    print "Max absolute difference is %.4f" % max_abs_diff
    print "Mean percent difference is %.4f%%" % mean_perc_diff
    print "Max percent difference is %.4f%%" % max_perc_diff
    if mean_abs_diff <= 0.1 and max_abs_diff <= 0.1:
        print "Acceptable."
    else:
        print "Difference unacceptable!"
    print '\n'

    
def main():
    test_data = np.genfromtxt("test_data.txt")
    test_uk_ok_produce_same_result(test_data)
    test_linear_drift()
    test1(test_data)
    test2(test_data)
    # test3(test_data)
                              
if __name__ == '__main__':
    main()
