__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    NumPy
    SciPy
    MatPlotLib

Classes:
    UniversalKriging: Provides greater control over 2D kriging by
        utilizing drift terms.

References:
    P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.
    
Copyright (C) 2014 Benjamin S. Murphy

This file is part of PyKrige.

PyKrige is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

PyKrige is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, go to <https://www.gnu.org/>.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import variogram_models


class UniversalKriging:
    """class UniversalKriging
    Provides greater control over 2D kriging by utilizing drift terms.

    Dependencies:
        NumPy
        SciPy
        MatPlotLib

    Inputs:
        X (array-like): X-coordinates of data points.
        Y (array-like): Y-coordinates of data points.
        Z (array-like): Values at data points.

        variogram_model (string, optional): Specified which variogram model to use;
            may be one of the following: linear, power, gaussian, spherical,
            exponential. Default is linear variogram model.
        variogram_parameters (array-like, optional): Parameters that define the
            specified variogram model. If not provided, parameters will be automatically
            calculated such that the root-mean-square error for the fit variogram
            function is minimized.
                linear - [slope, nugget]
                power - [scale, exponent, nugget]
                gaussian - [sill, range, nugget]
                spherical - [sill, range, nugget]
                exponential - [sill, range, nugget]
        nlags (int, optional): Number of averaging bins for the semivariogram.
            Default is 6.
        anisotropy_scaling (float, optional): Scalar stretching value to take
            into account anisotropy. Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction.
        anisotropy_angle (float, optional): Angle (in degrees) by which to
            rotate coordinate system in order to take into account anisotropy.
            Default is 0 (no rotation).

        drift_terms (list of strings, optional): List of drift terms to include in
            universal kriging. Supported drift terms are currently
            'regional_linear', 'point_log', and 'external_Z'.
        point_drift (array-like, optional): Array-like object that contains the
            coordinates and strengths of the point-logarithmic drift terms. Array
            shape must be Nx3, where N is the number of point drift terms. First
            column (index 0) must contain x-coordinates, second column (index 1)
            must contain y-coordinates, and third column (index 2) must contain
            the strengths of each point term. Strengths are relative, so only
            the relation of the values to each other matters.
        external_drift (array-like, optional): Gridded data used for the external
            Z scalar drift term. Must be dim MxN, where M is in the y-direction
            and N is in the x-direction. Grid spacing does not need to be constant.
            If grid spacing is not constant, must specify the grid cell sizes.
        external_drift_x (array-like, optional): X-coordinates for gridded
            external Z-scalar data. Must be dim M or Mx1 (where M is the number
            of grid cells in the x-direction). The coordinate is treated as
            the center of the cell.
        external_drift_y (array-like, optional): Y-coordinates for gridded
            external Z-scalar data. Must be dim N or Nx1 (where N is the
            number of grid cells in the y-direction). The coordinate is
            treated as the center of the cell.
        external_drift_xspacing (array-like, optional): Cell sizes in x-direction.
            Must be specified if external_drift array spacing is not constant.
        external_drift_yspacing (array-like, optional): Cell sizes in y-direction.
            Must be specified if external_drift array spacing is not constant.

        verbose (Boolean, optional): Enables program text output to monitor
            kriging process. Default is False (off).
        enable_plotting (Boolean, optional): Enables plotting to display
            variogram. Default is False (off).

    Callable Methods:
        display_variogram_model(): Displays semivariogram and variogram model.

        update_variogram_model(variogram_model, variogram_parameters=None, nlags=6,
            anisotropy_scaling=1.0, anisotropy_angle=0.0):
            Changes the variogram model and variogram parameters for
            the kriging system.
            Inputs:
                variogram_model (string): May be any of the variogram models
                    listed above.
                variogram_parameters (list, optional): List of variogram model
                    parameters, as listed above. If not provided, a best fit model
                    will be calculated as described above.
                nlags (int, optional): Number of averaging bins for the semivariogram.
                    Defualt is 6.
                anisotropy_scaling (float, optional): Scalar stretching value to
                    take into account anisotropy. Default is 1 (effectively no
                    stretching). Scaling is applied in the y-direction.
                anisotropy_angle (float, optional): Angle (in degrees) by which to
                    rotate coordinate system in order to take into account
                    anisotropy. Default is 0 (no rotation).

        switch_verbose(): Enables/disables program text output. No arguments.
        switch_plotting(): Enables/disable variogram plot display. No arguments.

        get_epsilon_residuals(): Returns the epsilon residuals of the
            variogram fit. No arguments.
        plot_epsilon_residuals(): Plots the epsilon residuals of the variogram
            fit in the order in which they were calculated. No arguments.

        get_statistics(): Returns the Q1, Q2, and cR statistics for the
            variogram fit (in that order). No arguments.

        print_statistics(): Prints out the Q1, Q2, and cR statistics for
            the variogram fit.

        execute(GRIDX, GRIDY): Calculates a kriged grid.
            Inputs:
                GRIDX (array-like): X-coordinates of grid.
                GRIDY (array-like): Y-coordinates of grid.
            Outputs:
                Z (numpy array): kriged grid
                SigmaSq (numpy array): variance

    References:
        P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
        (Cambridge University Press, 1997) 272 p.
    """

    UNBIAS = True  # This can be changed to remove the unbiasedness condition
                   # Really for testing purposes only...

    def __init__(self, x, y, z, variogram_model='linear',
                 variogram_parameters=None, nlags=6,
                 anisotropy_scaling=1.0, anisotropy_angle=0.0,
                 drift_terms=[None], point_drift=None, external_drift=None,
                 external_drift_x=None, external_drift_y=None,
                 external_drift_xspacing=None, external_drift_yspacing=None,
                 verbose=False, enable_plotting=False):

        # Code assumes 1D input arrays. Ensures that this is the case.
        self.X_ORIG = np.array(x).flatten()
        self.Y_ORIG = np.array(y).flatten()
        self.Z = np.array(z).flatten()

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print "Plotting Enabled\n"

        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG))/2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG))/2.0
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        if self.verbose:
            print "Adjusting data for anisotropy..."
        self.X_ADJUSTED, self.Y_ADJUSTED = \
            self._adjust_for_anisotropy(np.copy(self.X_ORIG),
                                        np.copy(self.Y_ORIG))

        self.variogram_model = variogram_model
        if self.variogram_model == 'linear':
            self.variogram_function = variogram_models.linear_variogram_model
        if self.variogram_model == 'power':
            self.variogram_function = variogram_models.power_variogram_model
        if self.variogram_model == 'gaussian':
            self.variogram_function = variogram_models.gaussian_variogram_model
        if self.variogram_model == 'spherical':
            self.variogram_function = variogram_models.spherical_variogram_model
        if self.variogram_model == 'exponential':
            self.variogram_function = variogram_models.exponential_variogram_model
        self.variogram_model_parameters = variogram_parameters
        if self.verbose:
            print "Initializing variogram model..."
        self._initialize_variogram_model(nlags)
        if self.verbose:
            if self.variogram_model == 'linear':
                print "Using '%s' Variogram Model" % 'linear'
                print "Slope:", self.variogram_model_parameters[0]
                print "Nugget:", self.variogram_model_parameters[1], '\n'
            elif self.variogram_model == 'power':
                print "Using '%s' Variogram Model" % 'power'
                print "Scale:", self.variogram_model_parameters[0]
                print "Exponent:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2], '\n'
            else:
                print "Using '%s' Variogram Model" % self.variogram_model
                print "Sill:", self.variogram_model_parameters[0]
                print "Range:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2]
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = self._find_statistics()
        self.Q1 = self._calcQ1(self.epsilon)
        self.Q2 = self._calcQ2(self.epsilon)
        self.cR = self._calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print "Q1 =", self.Q1
            print "Q2 =", self.Q2
            print "cR =", self.cR, '\n'

        if self.verbose:
            print "Initializing drift terms..."

        if 'regional_linear' in drift_terms:
            self.regional_linear_drift = True
            if self.verbose:
                print "Implementing regional linear drift."
        else:
            self.regional_linear_drift = False

        if 'external_Z' in drift_terms:
            if external_drift is None:
                raise ValueError("Must specify external Z drift terms.")
            if external_drift_x is None or external_drift_y is None:
                raise ValueError("Must specify coordinates of external Z drift terms.")
            self.external_Z_drift = True
            self.external_Z_array = np.array(external_drift)
            self.external_Z_array_x = np.array(external_drift_x).flatten()
            self.external_Z_array_y = np.array(external_drift_y).flatten()
            if np.unique(self.external_Z_array_x[1:] - self.external_Z_array_x[:-1]).size != 1:
                if external_drift_xspacing is None:
                    raise ValueError("X-coordinate spacing is not constant. "
                                     "Must provide X-coordinate grid size.")
                else:
                    self.external_Z_array_x_spacing = np.array(external_drift_xspacing).flatten()
            else:
                self.external_Z_array_x_spacing = np.zeros(self.external_Z_array_x.shape)
                self.external_Z_array_x_spacing.fill(
                    np.unique(self.external_Z_array_x[1:] - self.external_Z_array_x[:-1])[0])
            if np.unique(self.external_Z_array_y[1:] - self.external_Z_array_y[:-1]).size != 1:
                if external_drift_yspacing is None:
                    raise ValueError("Y-coordinate spacing is not constant. "
                                     "Must provide Y-coordinate grid size.")
                else:
                    self.external_Z_array_y_spacing = np.array(external_drift_yspacing).flatten()
            else:
                self.external_Z_array_y_spacing = np.zeros(self.external_Z_array_y.shape)
                self.external_Z_array_y_spacing.fill(
                    np.unique(self.external_Z_array_y[1:] - self.external_Z_array_y[:-1])[0])
            self.z_scalars = self._calculate_data_point_zscalars(self.X_ORIG,
                                                                 self.Y_ORIG)
            if self.verbose:
                print "Implementing external Z drift."
        else:
            self.external_Z_drift = False

        if 'point_log' in drift_terms:
            if point_drift is None:
                raise ValueError("Must specify location(s) and strength(s) of point drift terms.")
            self.point_log_drift = True
            self.point_log_array = np.atleast_2d(np.array(point_drift))
            if self.verbose:
                print "Implementing external point-logarithmic drift; " \
                      "number of points =", self.point_log_array.shape[0], '\n'
        else:
            self.point_log_drift = False

    def _adjust_for_anisotropy(self, x, y):
        # Adjusts data coordinates to take into account anisotropy.
        # Can also be used to take into account data scaling.

        x -= self.XCENTER
        y -= self.YCENTER
        xshape = x.shape
        yshape = y.shape
        x = x.flatten()
        y = y.flatten()

        coords = np.vstack((x, y))
        stretch = np.array([[1, 0], [0, self.anisotropy_scaling]])
        rotate = np.array([[np.cos(self.anisotropy_angle * np.pi/180.0),
                            np.sin(self.anisotropy_angle * np.pi/180.0)],
                         [- np.sin(self.anisotropy_angle * np.pi/180.0),
                            np.cos(self.anisotropy_angle * np.pi/180.0)]])
        rotated_coords = np.dot(stretch, np.dot(rotate, coords))
        x = rotated_coords[0, :].reshape(xshape)
        y = rotated_coords[1, :].reshape(yshape)
        x += self.XCENTER
        y += self.YCENTER

        return x, y

    def _calculate_data_point_zscalars(self, x, y):
        # Determines the Z-scalar values at the specified coordinates
        # for use when setting up the kriging matrix.
        # Currently, the Z scalar values are extracted from the input Z grid
        # exactly at the specified coordinates. This means that if the Z grid
        # resolution is finer than the resolution of the desired kriged grid,
        # there is no averaging of the scalar values to return an average
        # Z value for that cell in the kriged grid. Rather, the exact Z value
        # right at the coordinate is used.s

        z_scalars = np.zeros(x.shape)
        for n in range(z_scalars.size):
            xn = x[n]
            yn = y[n]
            # external_x_index = np.where((self.external_Z_array_x +
            #                              self.external_Z_array_x_spacing/2.0 >= xn) &
            #                             (self.external_Z_array_x -
            #                              self.external_Z_array_x_spacing/2.0 <= xn))[0][0]
            # external_y_index = np.where((self.external_Z_array_y +
            #                              self.external_Z_array_y_spacing/2.0 >= yn) &
            #                             (self.external_Z_array_y -
            #                              self.external_Z_array_y_spacing/2.0 <= yn))[0][0]
            # bilinear interpolation
            external_x2_index = np.amin(np.where(self.external_Z_array_x >= xn)[0])
            external_x1_index = np.amax(np.where(self.external_Z_array_x <= xn)[0])
            external_y2_index = np.amin(np.where(self.external_Z_array_y >= yn)[0])
            external_y1_index = np.amax(np.where(self.external_Z_array_y <= yn)[0])
            z = (self.external_Z_array[external_y1_index, external_x1_index] *
                 (self.external_Z_array_x[external_x2_index] - xn) *
                 (self.external_Z_array_y[external_y2_index] - yn) +
                 self.external_Z_array[external_y1_index, external_x2_index] *
                 (xn - self.external_Z_array_x[external_x1_index]) *
                 (self.external_Z_array_y[external_y2_index] - yn) +
                 self.external_Z_array[external_y2_index, external_x1_index] *
                 (self.external_Z_array_x[external_x2_index] - xn) *
                 (yn - self.external_Z_array_y[external_y1_index]) +
                 self.external_Z_array[external_y2_index, external_x2_index] *
                 (xn - self.external_Z_array_x[external_x1_index]) *
                 (yn - self.external_Z_array_y[external_y1_index])) / \
                ((self.external_Z_array_x[external_x2_index] -
                  self.external_Z_array_x[external_x1_index]) *
                 (self.external_Z_array_y[external_y2_index] -
                  self.external_Z_array_y[external_y1_index]))
            # print self.external_Z_array_x[external_x1_index], \
            #     self.external_Z_array_y[external_y1_index], \
            #     self.external_Z_array[external_x1_index, external_y1_index]
            # print self.external_Z_array_x[external_x1_index], \
            #     self.external_Z_array_y[external_y2_index], \
            #     self.external_Z_array[external_x1_index, external_y2_index]
            # print self.external_Z_array_x[external_x2_index], \
            #     self.external_Z_array_y[external_y1_index], \
            #     self.external_Z_array[external_x2_index, external_y1_index]
            # print self.external_Z_array_x[external_x2_index], \
            #     self.external_Z_array_y[external_y2_index], \
            #     self.external_Z_array[external_x2_index, external_y2_index]
            z_scalars[n] = z
            # z_scalars[n] = self.external_Z_array[external_y_index, external_x_index]

        return z_scalars

    def _initialize_variogram_model(self, nlags):
        # Initializes the variogram model for kriging according
        # to user specifications or to defaults.

        x1, x2 = np.meshgrid(self.X_ADJUSTED, self.X_ADJUSTED)
        y1, y2 = np.meshgrid(self.Y_ADJUSTED, self.Y_ADJUSTED)
        z1, z2 = np.meshgrid(self.Z, self.Z)

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        d = np.sqrt(dx**2 + dy**2)
        g = 0.5 * dz**2

        indices = np.indices(d.shape)
        d = d[(indices[0, :, :] > indices[1, :, :])]
        g = g[(indices[0, :, :] > indices[1, :, :])]

        # Bins are computed such that there are more at shorter lags.
        # This effectively weights smaller distances more highly in
        # determining the variogram. As Kitanidis points out, the variogram
        # fit to the data at smaller lag distances is more important.
        dmax = np.amax(d)
        dmin = np.amin(d)
        dd = dmax - dmin
        bins = [dd*(0.5**n) + dmin for n in range(nlags, 1, -1)]
        bins.insert(0, dmin)
        bins.append(dmax)

        lags = np.zeros((nlags, 1))
        semivariance = np.zeros((nlags, 1))

        for n in range(nlags):
            # This 'if... else...' statement ensures that there are data
            # in the bin so that numpy can actually find the mean. If we
            # don't test this first, then Python kicks out an annoying warning
            # message when there is an empty bin and we try to calculate the
            # mean.
            if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
                lags[n, 0] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
                semivariance[n, 0] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
            else:
                lags[n] = np.nan
                semivariance[n] = np.nan

        self.lags = lags[~np.isnan(semivariance)]
        self.semivariance = semivariance[~np.isnan(semivariance)]

        if self.variogram_model_parameters is not None:
            if self.variogram_model == 'linear' and \
                            len(self.variogram_model_parameters) != 2:
                raise ValueError("Exactly two parameters required "
                                 "for linear variogram model")
            elif self.variogram_model != 'linear' and \
                            len(self.variogram_model_parameters) != 3:
                raise ValueError("Exactly three parameters required "
                                 "for %s variogram model" % self.variogram_model)
        else:
            self.variogram_model_parameters = self._calculate_variogram_model(
                self.lags, self.semivariance)

    def _variogram_function_error(self, params, x, y):
        # Function used to in fitting of variogram model.
        # Returns RMSE between calculated fit and actual data.

        diff = self.variogram_function(params, x) - y
        rmse = np.sqrt(np.mean(diff**2))
        return rmse

    def _calculate_variogram_model(self, lags, semivariance):
        # Function that fits a variogram model when parameters
        # are not specified.

        if self.variogram_model == 'linear':
            x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
                  np.amin(semivariance)]
            bnds = ((0.0, 1000000000.0), (0.0, np.amax(semivariance)))
        elif self.variogram_model == 'power':
            x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
                  1.1, np.amin(semivariance)]
            bnds = ((0.0, 1000000000.0), (0.01, 1.99), (0.0, np.amax(semivariance)))
        else:
            x0 = [np.amax(semivariance), 0.5*np.amax(lags), np.amin(semivariance)]
            bnds = ((0.0, 10*np.amax(semivariance)), (0.0, np.amax(lags)), (0.0, np.amax(semivariance)))

        res = minimize(self._variogram_function_error, x0,
                       args=(lags, semivariance), method='SLSQP', bounds=bnds)

        return res.x

    def update_variogram_model(self, variogram_model,
                               variogram_parameters=None,
                               nlags=6, anisotropy_scaling=1.0,
                               anisotropy_angle=0.0):
        """Allows user to update variogram type and/or variogram model parameters."""

        if anisotropy_scaling != self.anisotropy_scaling or \
           anisotropy_angle != self.anisotropy_angle:
            if self.verbose:
                print "Adjusting data for anisotropy..."
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            self.X_ADJUSTED, self.Y_ADJUSTED = \
                self._adjust_for_anisotropy(np.copy(self.X_ORIG),
                                            np.copy(self.Y_ORIG))

        self.variogram_model = variogram_model
        if self.variogram_model == 'linear':
            self.variogram_function = self._linear_variogram_model
        if self.variogram_model == 'power':
            self.variogram_function = self._power_variogram_model
        if self.variogram_model == 'gaussian':
            self.variogram_function = self._gaussian_variogram_model
        if self.variogram_model == 'spherical':
            self.variogram_function = self._spherical_variogram_model
        if self.variogram_model == 'exponential':
            self.variogram_function = self._exponential_variogram_model
        self.variogram_model_parameters = variogram_parameters
        if self.verbose:
            print "Updating variogram mode..."
        self._initialize_variogram_model(nlags)
        if self.verbose:
            if self.variogram_model == 'linear':
                print "Using '%s' Variogram Model" % 'linear'
                print "Slope:", self.variogram_model_parameters[0]
                print "Nugget:", self.variogram_model_parameters[1], '\n'
            elif self.variogram_model == 'power':
                print "Using '%s' Variogram Model" % 'power'
                print "Scale:", self.variogram_model_parameters[0]
                print "Exponent:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2], '\n'
            else:
                print "Using '%s' Variogram Model" % self.variogram_model
                print "Sill:", self.variogram_model_parameters[0]
                print "Range:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2]
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = self._find_statistics()
        self.Q1 = self._calcQ1(self.epsilon)
        self.Q2 = self._calcQ2(self.epsilon)
        self.cR = self._calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print "Q1 =", self.Q1
            print "Q2 =", self.Q2
            print "cR =", self.cR, '\n'

    def display_variogram_model(self):
        """Displays variogram model with the actual binned data"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lags, self.semivariance, 'r*')
        ax.plot(self.lags,
                self.variogram_function(self.variogram_model_parameters, self.lags,),
                'k-')
        plt.show()

    def switch_verbose(self):
        """Allows user to switch code talk-back on/off. Takes no arguments."""
        self.verbose = not self.verbose

    def switch_plotting(self):
        """Allows user to switch plot display on/off. Takes no arguments."""
        self.enable_plotting = not self.enable_plotting

    def _krige_without_drifts(self, x, y, z, coords):
        # Sets up and solves the kriging matrix for the given coordinate pair.
        # Does not utilize drift terms.

        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2)

        n = x.shape[0]
        A = np.zeros((n+1, n+1))
        A[:n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
        np.fill_diagonal(A, 0)
        A[n, :] = 1
        A[:, n] = 1
        A[n, n] = 0

        b = np.zeros((n+1, 1))
        b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
        b[n, 0] = 1

        x = np.linalg.solve(A, b)
        zinterp = np.sum(x[:n, 0] * z)
        sigmasq = np.sum(x[:, 0] * -b[:, 0])

        return zinterp, sigmasq

    def _find_statistics(self):
        # Calculates variogram fit statistics.

        delta = np.zeros(self.Z.shape)
        sigma = np.zeros(self.Z.shape)

        for n in range(self.Z.shape[0]):
            if n == 0:
                delta[n] = 0.0
                sigma[n] = 0.0
            else:
                z, ss = self._krige_without_drifts(self.X_ADJUSTED[:n],
                                                   self.Y_ADJUSTED[:n], self.Z[:n],
                                                   (self.X_ADJUSTED[n], self.Y_ADJUSTED[n]))
                d = self.Z[n] - z
                delta[n] = d
                sigma[n] = np.sqrt(ss)

        delta = delta[1:]
        sigma = sigma[1:]
        epsilon = delta/sigma

        return delta, sigma, epsilon

    def _calcQ1(self, epsilon):
        return abs(np.sum(epsilon)/(epsilon.shape[0] - 1))

    def _calcQ2(self, epsilon):
        return np.sum(epsilon**2)/(epsilon.shape[0] - 1)

    def _calc_cR(self, Q2, sigma):
        return Q2 * np.exp(np.sum(np.log(sigma**2))/sigma.shape[0])

    def get_epsilon_residuals(self):
        # Returns the epsilon residuals for the variogram fit.
        return self.epsilon

    def plot_epsilon_residuals(self):
        # Plots the epsilon residuals for the variogram fit.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        plt.show()

    def get_statistics(self):
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        print "Q1 =", self.Q1
        print "Q2 =", self.Q2
        print "cR =", self.cR

    def _krige_with_drifts(self, x, y, z, coords):
        # Sets up and solves the kriging matrix for the given coordinate pair.
        # Utilizes drift terms.

        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2)

        n = x.shape[0]
        n_withdrifts = n
        if self.regional_linear_drift:
            n_withdrifts += 2
        if self.point_log_drift:
            n_withdrifts += self.point_log_array.shape[0]
        if self.external_Z_drift:
            n_withdrifts += 1
        if self.UNBIAS:
            A = np.zeros((n_withdrifts + 1, n_withdrifts + 1))
        else:
            A = np.zeros((n_withdrifts, n_withdrifts))
        A[:n, :n] = - self.variogram_function(self.variogram_model_parameters, d)

        np.fill_diagonal(A, 0.0)
        index = n
        if self.regional_linear_drift:
            A[:n, index] = x
            A[index, :n] = x
            index += 1
            A[:n, index] = y
            A[index, :n] = y
            index += 1
        if self.point_log_drift:
            for well_no in range(self.point_log_array.shape[0]):
                dist = np.sqrt((x - self.point_log_array[well_no, 0])**2 +
                               (y - self.point_log_array[well_no, 1])**2)
                A[:n, index] = - self.point_log_array[well_no, 2] * np.log(dist)
                A[index, :n] = - self.point_log_array[well_no, 2] * np.log(dist)
                index += 1
        if self.external_Z_drift:
            A[:n, index] = self.z_scalars
            A[index, :n] = self.z_scalars
            index += 1
        if index != n_withdrifts:
            print "WARNING: Error in creating kriging matrix. Kriging may fail."
        if self.UNBIAS:
            A[n_withdrifts, :n] = 1.0
            A[:n, n_withdrifts] = 1.0
            A[n:n_withdrifts + 1, n:n_withdrifts + 1] = 0.0

        if self.UNBIAS:
            b = np.zeros((n_withdrifts + 1, 1))
        else:
            b = np.zeros((n_withdrifts, 1))
        b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
        index = n
        if self.regional_linear_drift:
            b[index, 0] = coords[0]
            index += 1
            b[index, 0] = coords[1]
            index += 1
        if self.point_log_drift:
            for well_no in range(self.point_log_array.shape[0]):
                dist = np.sqrt((coords[0] - self.point_log_array[well_no, 0])**2 +
                               (coords[1] - self.point_log_array[well_no, 1])**2)
                b[index, 0] = - self.point_log_array[well_no, 2] * np.log(dist)
                index += 1
        if self.external_Z_drift:
            b[index, 0] = self._calculate_data_point_zscalars(np.array([coords[0]]),
                                                              np.array([coords[1]]))
            index += 1
        if index != n_withdrifts:
            print "WARNING: Error in setting up kriging system. Kriging may fail."
        if self.UNBIAS:
            b[n_withdrifts, 0] = 1.0

        # print A[-2:, :]

        x = np.linalg.solve(A, b)
        zinterp = np.sum(x[:n, 0] * z)
        sigmasq = np.sum(x[:, 0] * -b[:, 0])

        return zinterp, sigmasq

    def execute(self, gridx, gridy):
        """Calculates a kriged grid and the associated variance.

        Inputs:
            gridx (array-like, dim Nx1): X-coordinates of NxM grid.
            gridy (array-like, dim Mx1): Y-coordinates of NxM grid.
        Outputs:
            gridx (numpy array, dim MxN): Z-values of grid.
            sigmasq (numpy array, dim MxN): Variance at grid points.
        """

        if self.verbose:
            print "Executing Ordinary Kriging...\n"

        gridx = np.array(gridx).flatten()
        gridy = np.array(gridy).flatten()
        gridded_x, gridded_y = np.meshgrid(gridx, gridy)
        gridded_x, gridded_y = self._adjust_for_anisotropy(gridded_x, gridded_y)
        gridz = np.zeros((gridy.shape[0], gridx.shape[0]))
        sigmasq = np.zeros((gridy.shape[0], gridx.shape[0]))
        for m in range(gridz.shape[0]):
            for n in range(gridz.shape[1]):
                z, ss = self._krige_with_drifts(self.X_ADJUSTED, self.Y_ADJUSTED,
                                                self.Z, (gridded_x[m, n], gridded_y[m, n]))
                gridz[m, n] = z
                sigmasq[m, n] = ss

        return gridz, sigmasq
