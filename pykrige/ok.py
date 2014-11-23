__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    NumPy
    SciPy
    MatPlotLib

Classes:
    OrdinaryKriging: Convenience class for easy access to 2D Ordinary Kriging.

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
import matplotlib.pyplot as plt
import variogram_models
import core


class OrdinaryKriging:
    """class OrdinaryKriging
    Convenience class for easy access to 2D Ordinary Kriging

    Dependencies:
        NumPy
        MatPlotLib

    Inputs:
        X (array-like): X-coordinates of data points.
        Y (array-like): Y-coordinates of data points.
        Z (array-like): Values at data points.

        variogram_model (string, optional): Specified which variogram model to use;
            may be one of the following: linear, power, gaussian, spherical,
            exponential. Default is linear variogram model.
        variogram_parameters (list, optional): Parameters that define the
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
            the variogram fit. NOTE that ideally Q1 is close to zero,
            Q2 is close to 1, and cR is as small as possible.

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

    def __init__(self, x, y, z, variogram_model='linear',
                 variogram_parameters=None, nlags=6,
                 anisotropy_scaling=1.0, anisotropy_angle=0.0,
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
            core.adjust_for_anisotropy(np.copy(self.X_ORIG), np.copy(self.Y_ORIG),
                                       self.XCENTER, self.YCENTER,
                                       self.anisotropy_scaling, self.anisotropy_angle)

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
        if self.verbose:
            print "Initializing variogram model..."
        self.lags, self.semivariance, self.variogram_model_parameters = \
            core.initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags)
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
                print "Nugget:", self.variogram_model_parameters[2], '\n'
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = core.find_statistics(self.X_ADJUSTED,
                                                                    self.Y_ADJUSTED,
                                                                    self.Z,
                                                                    self.variogram_function,
                                                                    self.variogram_model_parameters)
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print "Q1 =", self.Q1
            print "Q2 =", self.Q2
            print "cR =", self.cR, '\n'

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
                core.adjust_for_anisotropy(np.copy(self.X_ORIG),
                                           np.copy(self.Y_ORIG),
                                           self.XCENTER, self.YCENTER,
                                           self.anisotropy_scaling,
                                           self.anisotropy_angle)

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
        if self.verbose:
            print "Updating variogram mode..."
        self.lags, self.semivariance, self.variogram_model_parameters = \
            core.initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags)
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
                print "Nugget:", self.variogram_model_parameters[2], '\n'
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = core.find_statistics(self.X_ADJUSTED,
                                                                    self.Y_ADJUSTED,
                                                                    self.Z,
                                                                    self.variogram_function,
                                                                    self.variogram_model_parameters)
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
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
                self.variogram_function(self.variogram_model_parameters, self.lags), 'k-')
        plt.show()

    def switch_verbose(self):
        """Allows user to switch code talk-back on/off. Takes no arguments."""
        self.verbose = not self.verbose

    def switch_plotting(self):
        """Allows user to switch plot display on/off. Takes no arguments."""
        self.enable_plotting = not self.enable_plotting

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

    def execute(self, gridx, gridy):
        """Calculates a kriged grid and the associated variance.

        Inputs:
            gridx (array-like, dim Nx1): X-coordinates of NxM grid.
            gridy (array-like, dim Mx1): Y-coordinates of NxM grid.
        Outputs:
            gridz (numpy array, dim NxM): Z-values of grid.
            sigmasq (numpy array, dim NxM): Variance at grid points.
        """

        if self.verbose:
            print "Executing Ordinary Kriging...\n"

        gridx = np.array(gridx).flatten()
        gridy = np.array(gridy).flatten()
        gridded_x, gridded_y = np.meshgrid(gridx, gridy)
        gridded_x, gridded_y = core.adjust_for_anisotropy(gridded_x, gridded_y,
                                                          self.XCENTER, self.YCENTER,
                                                          self.anisotropy_scaling,
                                                          self.anisotropy_angle)
        gridz = np.zeros((gridy.shape[0], gridx.shape[0]))
        sigmasq = np.zeros((gridy.shape[0], gridx.shape[0]))
        for m in range(gridz.shape[0]):
            for n in range(gridz.shape[1]):
                z, ss = core.krige(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                   (gridded_x[m, n], gridded_y[m, n]),
                                   self.variogram_function,
                                   self.variogram_model_parameters)
                gridz[m, n] = z
                sigmasq[m, n] = ss

        return gridz, sigmasq
