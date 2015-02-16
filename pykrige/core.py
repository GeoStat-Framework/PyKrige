__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy
    scipy (scipy.optimize.minimize())

Functions:
    adjust_for_anisotropy(x, y, xcenter, ycenter, scaling, angle):
        Returns X and Y arrays of adjusted data coordinates. Angle is CCW.
    initialize_variogram_model(x, y, z, variogram_model, variogram_model_parameters,
                               variogram_function, nlags):
        Returns lags, semivariance, and variogram model parameters as a list.
    variogram_function_error(params, x, y, variogram_function):
        Called by calculate_variogram_model.
    calculate_variogram_model(lags, semivariance, variogram_model, variogram_function):
        Returns variogram model parameters that minimize the RMSE between the specified
        variogram function and the actual calculated variogram points.
    krige(x, y, z, coords, variogram_function, variogram_model_parameters):
        Function that solves the ordinary kriging system for a single specified point.
        Returns the Z value and sigma squared for the specified coordinates.
    find_statistics(x, y, z, variogram_funtion, variogram_model_parameters):
        Returns the delta, sigma, and epsilon values for the variogram fit.
    calcQ1(epsilon):
        Returns the Q1 statistic for the variogram fit (see Kitanidis).
    calcQ2(epsilon):
        Returns the Q2 statistic for the variogram fit (see Kitanidis).
    calc_cR(Q2, sigma):
        Returns the cR statistic for the variogram fit (see Kitanidis).

References:
    P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015 Benjamin S. Murphy
"""

import numpy as np
from scipy.optimize import minimize


def adjust_for_anisotropy(x, y, xcenter, ycenter, scaling, angle):
    # Adjusts data coordinates to take into account anisotropy.
    # Can also be used to take into account data scaling.

    x -= xcenter
    y -= ycenter
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    coords = np.vstack((x, y))
    stretch = np.array([[1, 0], [0, scaling]])
    rotate = np.array([[np.cos(-angle * np.pi/180.0),
                        np.sin(-angle * np.pi/180.0)],
                     [- np.sin(-angle * np.pi/180.0),
                        np.cos(-angle * np.pi/180.0)]])
    rotated_coords = np.dot(stretch, np.dot(rotate, coords))
    x = rotated_coords[0, :].reshape(xshape)
    y = rotated_coords[1, :].reshape(yshape)
    x += xcenter
    y += ycenter

    return x, y


def initialize_variogram_model(x, y, z, variogram_model,
                               variogram_model_parameters,
                               variogram_function,
                               nlags, weight):
    # Initializes the variogram model for kriging according
    # to user specifications or to defaults.

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    d = np.sqrt(dx**2 + dy**2)
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
        elif variogram_model != 'linear' and len(variogram_model_parameters) != 3:
            raise ValueError("Exactly three parameters required "
                             "for %s variogram model" % variogram_model)
    else:
        variogram_model_parameters = calculate_variogram_model(lags, semivariance,
                                                               variogram_model, variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def variogram_function_error(params, x, y, variogram_function, weight):
    # Function used to in fitting of variogram model.
    # Returns RMSE between calculated fit and actual data.

    diff = variogram_function(params, x) - y

    if weight:
        weights = np.arange(x.size, 0.0, -1.0)
        weights /= np.sum(weights)
        rmse = np.sqrt(np.average(diff**2, weights=weights))
    else:
        rmse = np.sqrt(np.mean(diff**2))

    return rmse


def calculate_variogram_model(lags, semivariance, variogram_model, variogram_function, weight):
    # Function that fits a variogram model when parameters are not specified.

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


def krige(x, y, z, coords, variogram_function, variogram_model_parameters):
        # Sets up and solves the kriging matrix for the given coordinate pair.
        # This function is now only used for the statistics calculations.

        zero_index = None
        zero_value = False

        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2)
        if np.any(np.absolute(bd) <= 1e-10):
            zero_value = True
            zero_index = np.where(bd <= 1e-10)[0][0]

        n = x.shape[0]
        A = np.zeros((n+1, n+1))
        A[:n, :n] = - variogram_function(variogram_model_parameters, d)
        np.fill_diagonal(A, 0.0)
        A[n, :] = 1.0
        A[:, n] = 1.0
        A[n, n] = 0.0

        b = np.zeros((n+1, 1))
        b[:n, 0] = - variogram_function(variogram_model_parameters, bd)
        if zero_value:
            b[zero_index, 0] = 0.0
        b[n, 0] = 1.0

        x = np.linalg.solve(A, b)
        zinterp = np.sum(x[:n, 0] * z)
        sigmasq = np.sum(x[:, 0] * -b[:, 0])

        return zinterp, sigmasq


def find_statistics(x, y, z, variogram_function, variogram_model_parameters):
    # Calculates variogram fit statistics.

    delta = np.zeros(z.shape)
    sigma = np.zeros(z.shape)

    for n in range(z.shape[0]):
        if n == 0:
            delta[n] = 0.0
            sigma[n] = 0.0
        else:
            z_, ss_ = krige(x[:n], y[:n], z[:n], (x[n], y[n]),
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