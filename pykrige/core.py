from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy
    scipy (scipy.optimize.minimize())

Functions:
    _adjust_for_anisotropy(X, y, center, scaling, angle):
        Returns X_adj array of adjusted data coordinates. Angles are CCW about
        specified axes. Scaling is applied in rotated coordinate system.
    initialize_variogram_model(x, y, z, variogram_model, variogram_model_parameters,
                               variogram_function, nlags):
        Returns lags, semivariance, and variogram model parameters as a list.
    initialize_variogram_model_3d(x, y, z, values, variogram_model,
                                  variogram_model_parameters, variogram_function, nlags):
        Returns lags, semivariance, and variogram model parameters as a list.
    variogram_function_error(params, x, y, variogram_function):
        Called by calculate_variogram_model.
    calculate_variogram_model(lags, semivariance, variogram_model, variogram_function):
        Returns variogram model parameters that minimize the RMSE between the specified
        variogram function and the actual calculated variogram points.
    krige(x, y, z, coords, variogram_function, variogram_model_parameters):
        Function that solves the ordinary kriging system for a single specified point.
        Returns the Z value and sigma squared for the specified coordinates.
    krige_3d(x, y, z, vals, coords, variogram_function, variogram_model_parameters):
        Function that solves the ordinary kriging system for a single specified point.
        Returns the interpolated value and sigma squared for the specified coordinates.
    find_statistics(x, y, z, variogram_funtion, variogram_model_parameters):
        Returns the delta, sigma, and epsilon values for the variogram fit.
    calcQ1(epsilon):
        Returns the Q1 statistic for the variogram fit (see Kitanidis).
    calcQ2(epsilon):
        Returns the Q2 statistic for the variogram fit (see Kitanidis).
    calc_cR(Q2, sigma):
        Returns the cR statistic for the variogram fit (see Kitanidis).
    great_circle_distance(lon1, lat1, lon2, lat2):
        Returns the great circle distance between two arrays of points given in spherical
        coordinates. Spherical coordinates are expected in degrees. Angle definition
        follows standard longitude/latitude definition.

References:
[1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

[2] T. Vincenty, Direct and Inverse Solutions of Geodesics on the Ellipsoid
    with Application of Nested Equations, Survey Review 23 (176),
    (Directorate of Overseas Survey, Kingston Road, Tolworth, Surrey 1975)

Copyright (c) 2015 Benjamin S. Murphy
"""

import numpy as np
from scipy.optimize import minimize


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between one or multiple
    pairs of points on a unit sphere.

    Parameters:
    -----------
    lon1: float scalar or numpy array
        Longitude coordinate(s) of the first element(s) of the point
        pair(s), given in degrees.
    lat1: float scalar or numpy array
        Latitude coordinate(s) of the first element(s) of the point
        pair(s), given in degrees.
    lon2: float scalar or numpy array
        Longitude coordinate(s) of the second element(s) of the point
        pair(s), given in degrees.
    lat2: float scalar or numpy array
        Latitude coordinate(s) of the second element(s) of the point
        pair(s), given in degrees.

    Calculation of distances follows numpy elementwise semantics, so if
    an array of length N is passed, all input parameters need to be
    arrays of length N or scalars.


    Returns:
    --------
    distance: float
              The great circle distance(s) (in degrees) between the
              given pair(s) of points.

    """
    # Convert to radians:
    lat1 = np.array(lat1)*np.pi/180.0
    lat2 = np.array(lat2)*np.pi/180.0
    dlon = (lon1-lon2)*np.pi/180.0

    # Evaluate trigonometric functions that need to be evaluated more
    # than once:
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    cd = np.cos(dlon)

    # This uses the arctan version of the great-circle distance function
    # from en.wikipedia.org/wiki/Great-circle_distance for increased
    # numerical stability.
    # Formula can be obtained from [2] combining eqns. (14)-(16)
    # for spherical geometry (f=0).

    return 180.0/np.pi*np.arctan2(
               np.sqrt((c2*np.sin(dlon))**2 +
                       (c1*s2-s1*c2*cd)**2),
               s1*s2+c1*c2*cd)

def euclid3_to_great_circle(euclid3_distance):
    """
    Convert euclidean distance between points on a unit sphere to
    the corresponding great circle distance.


    Parameters:
    -----------
    euclid3_distance: float scalar or numpy array
        The euclidean three-space distance(s) between points on a
        unit sphere, thus between [0,2].


    Returns:
    --------
    great_circle_dist: float scalar or numpy array
        The corresponding great circle distance(s) between the
        points.
    """
    # Eliminate some possible numerical errors:
    euclid3_distance[euclid3_distance>2.0] = 2.0
    return 180.0 - 360.0/np.pi*np.arccos(0.5*euclid3_distance)


def _adjust_for_anisotropy(X, center, scaling, angle):
    """Adjusts data coordinates to take into account anisotropy.
    Can also be used to take into account data scaling.

    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    center: ndarray
        float array [n_dim], the coordinate of centers
    scaling: ndarray
        float array [n_dim - 1], the scaling of last two dimensions
    angle : ndarray
        float array [2*n_dim - 3], the anysotropy angle (degrees)

    Returns
    -------
    X_adj : ndarray
        float array [n_samples, n_dim], the X array adjusted for anisotropy.
    """

    center = np.asarray(center)[None, :]
    angle = np.asarray(angle)*np.pi/180

    X -= center

    Ndim = X.shape[1]

    if Ndim == 1:
        raise NotImplementedError('Not implemnented yet?')
    elif Ndim == 2:
        stretch = np.array([[1, 0], [0, scaling[0]]])
        rot_tot = np.array([[np.cos(-angle[0]), -np.sin(-angle[0])],
                           [np.sin(-angle[0]), np.cos(-angle[0])]])
    elif Ndim == 3:
        stretch = np.array([[1., 0., 0.], [0., scaling[0], 0.], [0., 0., scaling[1]]])
        rotate_x = np.array([[1., 0., 0.],
                             [0., np.cos(-angle[0]), -np.sin(-angle[0])],
                             [0., np.sin(-angle[0]), np.cos(-angle[0])]])
        rotate_y = np.array([[np.cos(-angle[1]), 0., np.sin(-angle[1])],
                             [0., 1., 0.],
                             [-np.sin(-angle[1]), 0., np.cos(-angle[1])]])
        rotate_z = np.array([[np.cos(-angle[2]), -np.sin(-angle[2]), 0.],
                             [np.sin(-angle[2]), np.cos(-angle[2]), 0.],
                             [0., 0., 1.]])
        rot_tot = np.dot(rotate_z, np.dot(rotate_y, rotate_x))
    else:
        raise ValueError("Adjust for anysotropy function doesn't support ND spaces where N>3")
    X_adj = np.dot(stretch, np.dot(rot_tot, X.T)).T

    X_adj += center

    return X_adj


def initialize_variogram_model(x, y, z, variogram_model, variogram_model_parameters,
                               variogram_function, nlags, weight, coordinates_type):
    """Initializes the variogram model for kriging according
    to user specifications or to defaults"""

    x1, x2 = np.meshgrid(x, x, sparse=True)
    y1, y2 = np.meshgrid(y, y, sparse=True)
    z1, z2 = np.meshgrid(z, z, sparse=True)
    dz = z1 - z2

    if coordinates_type == 'euclidean':
        dx = x1 - x2
        dy = y1 - y2
        d = np.sqrt(dx**2 + dy**2)
    elif coordinates_type == 'geographic':
        # Assume x => lon, y => lat
        d = great_circle_distance(x1, y1, x2, y2)

    g = 0.5 * dz**2

    indices = np.indices(d.shape)
    d = d[(indices[0, :, :] > indices[1, :, :])]
    g = g[(indices[0, :, :] > indices[1, :, :])]

    # Equal-sized bins are now implemented. The upper limit on the bins
    # is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities
    # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
    # Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin)/nlags
    bins = [dmin + n*dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    # This old binning method was experimental and doesn't seem
    # to work too well. Bins were computed such that there are more
    # at shorter lags. This effectively weights smaller distances more
    # highly in determining the variogram. As Kitanidis points out,
    # the variogram fit to the data at smaller lag distances is more
    # important. However, the value at the largest lag probably ends up
    # being biased too high for the larger values and thereby throws off
    # automatic variogram calculation and confuses comparison of the
    # semivariogram with the variogram model.
    #
    # dmax = np.amax(d)
    # dmin = np.amin(d)
    # dd = dmax - dmin
    # bins = [dd*(0.5**n) + dmin for n in range(nlags, 1, -1)]
    # bins.insert(0, dmin)
    # bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):
        # This 'if... else...' statement ensures that there are data
        # in the bin so that numpy can actually find the mean. If we
        # don't test this first, then Python kicks out an annoying warning
        # message when there is an empty bin and we try to calculate the mean.
        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    if variogram_model_parameters is not None:
        if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
            raise ValueError("Exactly two parameters required "
                             "for linear variogram model")
        elif (variogram_model == 'power' or variogram_model == 'spherical' or variogram_model == 'exponential'
              or variogram_model == 'gaussian') and len(variogram_model_parameters) != 3:
            raise ValueError("Exactly three parameters required "
                             "for %s variogram model" % variogram_model)
    else:
        if variogram_model == 'custom':
            raise ValueError("Variogram parameters must be specified when implementing custom variogram model.")
        else:
            variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
                                                                   variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def initialize_variogram_model_3d(x, y, z, values, variogram_model, variogram_model_parameters,
                                  variogram_function, nlags, weight):
    """Initializes the variogram model for kriging according
    to user specifications or to defaults"""

    x1, x2 = np.meshgrid(x, x, sparse=True)
    y1, y2 = np.meshgrid(y, y, sparse=True)
    z1, z2 = np.meshgrid(z, z, sparse=True)
    val1, val2 = np.meshgrid(values, values)
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    g = 0.5 * (val1 - val2)**2

    indices = np.indices(d.shape)
    d = d[(indices[0, :, :] > indices[1, :, :])]
    g = g[(indices[0, :, :] > indices[1, :, :])]

    # The upper limit on the bins is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities (specifically, say, ending up as
    # 0.99999999999999 instead of 1.0). Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin)/nlags
    bins = [dmin + n*dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):
        # This 'if... else...' statement ensures that there are data in the bin so that numpy can actually
        # find the mean. If we don't test this first, then Python kicks out an annoying warning message
        # when there is an empty bin and we try to calculate the mean.
        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    if variogram_model_parameters is not None:
        if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
            raise ValueError("Exactly two parameters required "
                             "for linear variogram model")
        elif (variogram_model == 'power' or variogram_model == 'spherical' or variogram_model == 'exponential'
              or variogram_model == 'gaussian') and len(variogram_model_parameters) != 3:
            raise ValueError("Exactly three parameters required "
                             "for %s variogram model" % variogram_model)
    else:
        if variogram_model == 'custom':
            raise ValueError("Variogram parameters must be specified when implementing custom variogram model.")
        else:
            variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
                                                                   variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def variogram_function_error(params, x, y, variogram_function, weight):
    """Function used to in fitting of variogram model.
    Returns RMSE between calculated fit and actual data."""

    diff = variogram_function(params, x) - y

    if weight:
        weights = np.arange(x.size, 0.0, -1.0)
        weights /= np.sum(weights)
        rmse = np.sqrt(np.average(diff**2, weights=weights))
    else:
        rmse = np.sqrt(np.mean(diff**2))

    return rmse


def calculate_variogram_model(lags, semivariance, variogram_model, variogram_function, weight):
    """Function that fits a variogram model when parameters are not specified."""

    if variogram_model == 'linear':
        x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
              np.amin(semivariance)]
        bnds = ((0.0, 1000000000.0), (0.0, np.amax(semivariance)))
    elif variogram_model == 'power':
        x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
              1.1, np.amin(semivariance)]
        bnds = ((0.0, 1000000000.0), (0.01, 1.99), (0.0, np.amax(semivariance)))
    else:
        x0 = [np.amax(semivariance), 0.5*np.amax(lags), np.amin(semivariance)]
        bnds = ((0.0, 10*np.amax(semivariance)), (0.0, np.amax(lags)), (0.0, np.amax(semivariance)))

    res = minimize(variogram_function_error, x0, args=(lags, semivariance, variogram_function, weight),
                   method='SLSQP', bounds=bnds)

    return res.x


def krige(x, y, z, coords, variogram_function, variogram_model_parameters, coordinates_type):
        """Sets up and solves the kriging matrix for the given coordinate pair.
        This function is now only used for the statistics calculations."""

        zero_index = None
        zero_value = False

        x1, x2 = np.meshgrid(x, x, sparse=True)
        y1, y2 = np.meshgrid(y, y, sparse=True)

        if coordinates_type == 'euclidean':
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2)
        elif coordinates_type == 'geographic':
            d = great_circle_distance(x1, y1, x2, y2)
            bd = great_circle_distance(x, y, coords[0]*np.ones(x.shape),
                                       coords[1]*np.ones(y.shape))
        if np.any(np.absolute(bd) <= 1e-10):
            zero_value = True
            zero_index = np.where(bd <= 1e-10)[0][0]

        n = x.shape[0]
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - variogram_function(variogram_model_parameters, d)
        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        b = np.zeros((n+1, 1))
        b[:n, 0] = - variogram_function(variogram_model_parameters, bd)
        if zero_value:
            b[zero_index, 0] = 0.0
        b[n, 0] = 1.0

        x_ = np.linalg.solve(a, b)
        zinterp = np.sum(x_[:n, 0] * z)
        sigmasq = np.sum(x_[:, 0] * -b[:, 0])

        return zinterp, sigmasq


def krige_3d(x, y, z, vals, coords, variogram_function, variogram_model_parameters):
        """Sets up and solves the kriging matrix for the given coordinate pair.
        This function is now only used for the statistics calculations."""

        zero_index = None
        zero_value = False

        x1, x2 = np.meshgrid(x, x, sparse=True)
        y1, y2 = np.meshgrid(y, y, sparse=True)
        z1, z2 = np.meshgrid(z, z, sparse=True)
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2 + (z - coords[2])**2)
        if np.any(np.absolute(bd) <= 1e-10):
            zero_value = True
            zero_index = np.where(bd <= 1e-10)[0][0]

        n = x.shape[0]
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - variogram_function(variogram_model_parameters, d)
        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        b = np.zeros((n+1, 1))
        b[:n, 0] = - variogram_function(variogram_model_parameters, bd)
        if zero_value:
            b[zero_index, 0] = 0.0
        b[n, 0] = 1.0

        x_ = np.linalg.solve(a, b)
        zinterp = np.sum(x_[:n, 0] * vals)
        sigmasq = np.sum(x_[:, 0] * -b[:, 0])

        return zinterp, sigmasq


def find_statistics(x, y, z, variogram_function, variogram_model_parameters, coordinates_type):
    """Calculates variogram fit statistics."""

    delta = np.zeros(z.shape)
    sigma = np.zeros(z.shape)

    for n in range(z.shape[0]):
        if n == 0:
            delta[n] = 0.0
            sigma[n] = 0.0
        else:
            z_, ss_ = krige(x[:n], y[:n], z[:n], (x[n], y[n]), variogram_function,
                            variogram_model_parameters, coordinates_type)
            d = z[n] - z_
            delta[n] = d
            sigma[n] = np.sqrt(ss_)

    delta = delta[1:]
    sigma = sigma[1:]
    epsilon = delta/sigma

    return delta, sigma, epsilon


def find_statistics_3d(x, y, z, vals, variogram_function, variogram_model_parameters):
    """Calculates variogram fit statistics for 3D problems."""

    delta = np.zeros(vals.shape)
    sigma = np.zeros(vals.shape)

    for n in range(z.shape[0]):
        if n == 0:
            delta[n] = 0.0
            sigma[n] = 0.0
        else:
            z_, ss_ = krige_3d(x[:n], y[:n], z[:n], vals[:n], (x[n], y[n], z[n]),
                               variogram_function, variogram_model_parameters)
            d = z[n] - z_
            delta[n] = d
            sigma[n] = np.sqrt(ss_)

    delta = delta[1:]
    sigma = sigma[1:]
    epsilon = delta/sigma

    return delta, sigma, epsilon


def calcQ1(epsilon):
    return abs(np.sum(epsilon)/(epsilon.shape[0] - 1))


def calcQ2(epsilon):
    return np.sum(epsilon**2)/(epsilon.shape[0] - 1)


def calc_cR(Q2, sigma):
    return Q2 * np.exp(np.sum(np.log(sigma**2))/sigma.shape[0])
