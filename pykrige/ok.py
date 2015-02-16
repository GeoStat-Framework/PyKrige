__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy
    scipy
    matplotlib

Classes:
    OrdinaryKriging: Convenience class for easy access to 2D Ordinary Kriging.

References:
    P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015 Benjamin S. Murphy
"""

import numpy as np
import matplotlib.pyplot as plt
import variogram_models
import core


class OrdinaryKriging:
    """class OrdinaryKriging
    Convenience class for easy access to 2D Ordinary Kriging

    Dependencies:
        numpy
        matplotlib

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
        weight (boolean, optional): Flag that specifies if semivariance at smaller lags
            should be weighted more heavily when automatically calculating variogram model.
            True indicates that weights will be applied. Default is False.
            (Kitanidis suggests that the values at smaller lags are more important in
            fitting a variogram model, so the option is provided to enable such weighting.)
        anisotropy_scaling (float, optional): Scalar stretching value to take
            into account anisotropy. Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction in the rotated data frame
            (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
            is not 0).
        anisotropy_angle (float, optional): CCW angle (in degrees) by which to
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
                weight (boolean, optional): Flag that specifies if semivariance at smaller lags
                    should be weighted more heavily when automatically calculating variogram model.
                    True indicates that weights will be applied. Default is False.
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

        execute(style, xpoints, ypoints, mask=None): Calculates a kriged grid.
            Inputs:
                style (string): Specifies how to treat input kriging points.
                    Specifying 'grid' treats xpoints and ypoints as two arrays of
                    x and y coordinates that define a rectangular grid.
                    Specifying 'points' treats xpoints and ypoints as two arrays
                    that provide coordinate pairs at which to solve the kriging system.
                    Specifying 'masked' treats xpoints and ypoints as two arrays of
                    x and y coordinates that define a rectangular grid and uses mask
                    to only evaluate specific points in the grid.
                xpoints (array-like, dim Nx1): If style is specific as 'grid' or 'masked',
                    x-coordinates of MxN grid. If style is specified as 'points',
                    x-coordinates of specific points at which to solve kriging system.
                ypoints (array-like, dim Mx1): If style is specified as 'grid' or 'masked',
                    y-coordinates of MxN grid. If style is specified as 'points',
                    y-coordinates of specific points at which to solve kriging system.
                mask (boolean array, dim MxN, optional): Specifies the points in the rectangular
                    grid defined by xpoints and ypoints that are to be excluded in the
                    kriging calculations. Must be provided if style is specified as 'masked'.
                    False indicates that the point should not be masked; True indicates that
                    the point should be masked.
            Outputs:
                zvalues (numpy array, dim MxN or dim Nx1): Z-values of specified grid or at the
                    specified set of points. If style was specified as 'masked', zvalues will
                    be a numpy masked array.
                sigmasq (numpy array, dim MxN or dim Nx1): Variance at specified grid points or
                    at the specified set of points. If style was specified as 'masked', sigmasq
                    will be a numpy masked array.

    References:
        P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
        (Cambridge University Press, 1997) 272 p.
    """

    eps = 1e-10   # Cutoff for comparison to zero

    def __init__(self, x, y, z, variogram_model='linear',
                 variogram_parameters=None, nlags=6, weight=False,
                 anisotropy_scaling=1.0, anisotropy_angle=0.0,
                 verbose=False, enable_plotting=False):

        # Code assumes 1D input arrays. Ensures that this is the case.
        # Copies are created to avoid any problems with referencing
        # the original passed arguments.
        self.X_ORIG = np.array(x, copy=True).flatten()
        self.Y_ORIG = np.array(y, copy=True).flatten()
        self.Z = np.array(z, copy=True).flatten()

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
        elif self.variogram_model == 'power':
            self.variogram_function = variogram_models.power_variogram_model
        elif self.variogram_model == 'gaussian':
            self.variogram_function = variogram_models.gaussian_variogram_model
        elif self.variogram_model == 'spherical':
            self.variogram_function = variogram_models.spherical_variogram_model
        elif self.variogram_model == 'exponential':
            self.variogram_function = variogram_models.exponential_variogram_model
        else:
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        if self.verbose:
            print "Initializing variogram model..."
        self.lags, self.semivariance, self.variogram_model_parameters = \
            core.initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags, weight)
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

    def update_variogram_model(self, variogram_model, variogram_parameters=None,
                               nlags=6, weight=False,
                               anisotropy_scaling=1.0, anisotropy_angle=0.0):
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
        elif self.variogram_model == 'power':
            self.variogram_function = variogram_models.power_variogram_model
        elif self.variogram_model == 'gaussian':
            self.variogram_function = variogram_models.gaussian_variogram_model
        elif self.variogram_model == 'spherical':
            self.variogram_function = variogram_models.spherical_variogram_model
        elif self.variogram_model == 'exponential':
            self.variogram_function = variogram_models.exponential_variogram_model
        else:
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        if self.verbose:
            print "Updating variogram mode..."
        self.lags, self.semivariance, self.variogram_model_parameters = \
            core.initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags, weight)
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

    def execute(self, style, xpoints, ypoints, mask=None):
        """Calculates a kriged grid and the associated variance.

        This is now the method that performs the main kriging calculation. Note that currently
        measurements (i.e., z values) are considered 'exact'. This means that, when a specified
        coordinate for interpolation is exactly the same as one of the data points, the variogram
        evaluated at the point is forced to be zero. Also, the diagonal of the kriging matrix is
        also always forced to be zero. In forcing the variogram evaluated at data points to be zero,
        we are effectively saying that there is no variance at that point (no uncertainty,
        so the value is 'exact').

        In the future, the code may include an extra 'exact_values' boolean flag that can be
        adjusted to specify whether to treat the measurements as 'exact'. Setting the flag
        to false would indicate that the variogram should not be forced to be zero at zero distance
        (i.e., when evaluated at data points). Instead, the uncertainty in the point will be
        equal to the nugget. This would mean that the diagonal of the kriging matrix would be set to
        the nugget instead of to zero.

        Inputs:
            style (string): Specifies how to treat input kriging points.
                Specifying 'grid' treats xpoints and ypoints as two arrays of
                x and y coordinates that define a rectangular grid.
                Specifying 'points' treats xpoints and ypoints as two arrays
                that provide coordinate pairs at which to solve the kriging system.
                Specifying 'masked' treats xpoints and ypoints as two arrays of
                x and y coordinates that define a rectangular grid and uses mask
                to only evaluate specific points in the grid.
            xpoints (array-like, dim Nx1): If style is specific as 'grid' or 'masked',
                x-coordinates of MxN grid. If style is specified as 'points',
                x-coordinates of specific points at which to solve kriging system.
            ypoints (array-like, dim Mx1): If style is specified as 'grid' or 'masked',
                y-coordinates of MxN grid. If style is specified as 'points',
                y-coordinates of specific points at which to solve kriging system.
                Note that in this case, xpoints and ypoints must have the same dimensions
                (i.e., M = N).
            mask (boolean array, dim MxN, optional): Specifies the points in the rectangular
                grid defined by xpoints and ypoints that are to be excluded in the
                kriging calculations. Must be provided if style is specified as 'masked'.
                False indicates that the point should not be masked, so that the kriging system
                is solved at the point; True indicates that the point should be masked,
                so that the kriging system is not solved at the point.
        Outputs:
            zvalues (numpy array, dim MxN or dim Nx1): Z-values of specified grid or at the
                specified set of points. If style was specified as 'masked', zvalues will
                be a numpy masked array.
            sigmasq (numpy array, dim MxN or dim Nx1): Variance at specified grid points or
                at the specified set of points. If style was specified as 'masked', sigmasq
                will be a numpy masked array.
        """

        if self.verbose:
            print "Executing Ordinary Kriging...\n"

        xpoints = np.array(xpoints, copy=True).flatten()
        ypoints = np.array(ypoints, copy=True).flatten()
        nx = xpoints.shape[0]
        ny = ypoints.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        if style == 'grid':

            grid_x, grid_y = np.meshgrid(xpoints, ypoints)
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y,
                                                        self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            x1, x2 = np.meshgrid(self.X_ADJUSTED, self.X_ADJUSTED)
            y1, y2 = np.meshgrid(self.Y_ADJUSTED, self.Y_ADJUSTED)
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            a = np.zeros((ny, nx, n+1, n+1))
            a[:, :, :n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
            index_grid = np.indices((ny, nx, n+1, n+1))
            a[index_grid[2] == index_grid[3]] = 0.0
            a[:, :, n, :] = 1.0
            a[:, :, :, n] = 1.0
            a[:, :, n, n] = 0.0

            grid_x_3d = np.repeat(grid_x[:, :, np.newaxis], n, axis=2)
            grid_y_3d = np.repeat(grid_y[:, :, np.newaxis], n, axis=2)
            data_x_3d = np.repeat(np.repeat(self.X_ADJUSTED[np.newaxis, np.newaxis, :], ny, axis=0), nx, axis=1)
            data_y_3d = np.repeat(np.repeat(self.Y_ADJUSTED[np.newaxis, np.newaxis, :], ny, axis=0), nx, axis=1)
            bd = np.sqrt((data_x_3d - grid_x_3d)**2 + (data_y_3d - grid_y_3d)**2)
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            b = np.zeros((ny, nx, n+1, 1))
            b[:, :, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], zero_index[2], 0] = 0.0
            b[:, :, n, 0] = 1.0

            x = np.linalg.solve(a, b)
            zvalues = np.sum(x[:, :, :n, 0] * self.Z, axis=2)
            sigmasq = np.sum(x[:, :, :, 0] * -b[:, :, :, 0], axis=2)

        elif style == 'masked':

            if mask is None:
                raise IOError("Must specify boolean masking array.")
            if mask.shape[0] != ny or mask.shape[1] != nx:
                if mask.shape[0] == nx and mask.shape[1] == ny:
                    mask = mask.T
                else:
                    raise ValueError("Mask dimensions do not match specified grid dimensions.")

            grid_x, grid_y = np.meshgrid(xpoints, ypoints)
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y,
                                                        self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            x1, x2 = np.meshgrid(self.X_ADJUSTED, self.X_ADJUSTED)
            y1, y2 = np.meshgrid(self.Y_ADJUSTED, self.Y_ADJUSTED)
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            a = np.zeros((ny, nx, n+1, n+1))
            a[:, :, :n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
            index_grid = np.indices((ny, nx, n+1, n+1))
            a[index_grid[2] == index_grid[3]] = 0.0
            a[:, :, n, :] = 1.0
            a[:, :, :, n] = 1.0
            a[:, :, n, n] = 0.0
            mask_a = np.repeat(np.repeat(mask[:, :, np.newaxis, np.newaxis], n+1, axis=2), n+1, axis=3)
            a = np.ma.array(a, mask=mask_a)

            grid_x_3d = np.repeat(grid_x[:, :, np.newaxis], n, axis=2)
            grid_y_3d = np.repeat(grid_y[:, :, np.newaxis], n, axis=2)
            data_x_3d = np.repeat(np.repeat(self.X_ADJUSTED[np.newaxis, np.newaxis, :], ny, axis=0), nx, axis=1)
            data_y_3d = np.repeat(np.repeat(self.Y_ADJUSTED[np.newaxis, np.newaxis, :], ny, axis=0), nx, axis=1)
            bd = np.sqrt((data_x_3d - grid_x_3d)**2 + (data_y_3d - grid_y_3d)**2)
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            b = np.zeros((ny, nx, n+1, 1))
            b[:, :, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], zero_index[2], 0] = 0.0
            b[:, :, n, 0] = 1.0
            mask_b = np.repeat(mask[:, :, np.newaxis, np.newaxis], n+1, axis=2)
            b = np.ma.array(b, mask=mask_b)

            x = np.linalg.solve(a, b)
            zvalues = np.sum(x[:, :, :n, 0] * self.Z, axis=2)
            sigmasq = np.sum(x[:, :, :, 0] * -b[:, :, :, 0], axis=2)

        elif style == 'points':
            if xpoints.shape != ypoints.shape:
                raise ValueError("xpoints and ypoints must have same dimensions "
                                 "when treated as listing discrete points.")

            xpoints, ypoints = core.adjust_for_anisotropy(xpoints, ypoints, self.XCENTER, self.YCENTER,
                                                          self.anisotropy_scaling, self.anisotropy_angle)

            x1, x2 = np.meshgrid(self.X_ADJUSTED, self.X_ADJUSTED)
            y1, y2 = np.meshgrid(self.Y_ADJUSTED, self.Y_ADJUSTED)
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            a = np.zeros((nx, n+1, n+1))
            a[:, :n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
            index_grid = np.indices((nx, n+1, n+1))
            a[index_grid[1] == index_grid[2]] = 0.0
            a[:, n, :] = 1.0
            a[:, :, n] = 1.0
            a[:, n, n] = 0.0

            x_vals = np.repeat(xpoints[:, np.newaxis], n, axis=1)
            y_vals = np.repeat(ypoints[:, np.newaxis], n, axis=1)
            x_data = np.repeat(self.X_ADJUSTED[np.newaxis, :], nx, axis=0)
            y_data = np.repeat(self.Y_ADJUSTED[np.newaxis, :], nx, axis=0)
            bd = np.sqrt((x_data - x_vals)**2 + (y_data - y_vals)**2)
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            b = np.zeros((nx, n+1, 1))
            b[:, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], 0] = 0.0
            b[:, n, 0] = 1.0

            x = np.linalg.solve(a, b)
            zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
            sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        return zvalues, sigmasq
