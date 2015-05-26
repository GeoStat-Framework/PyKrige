__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy
    scipy
    matplotlib

Classes:
    UniversalKriging: Provides greater control over 2D kriging by
        utilizing drift terms.

References:
    P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.
    
Copyright (c) 2015 Benjamin S. Murphy
"""

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import variogram_models
import core


class UniversalKriging:
    """class UniversalKriging
    Provides greater control over 2D kriging by utilizing drift terms.

    Dependencies:
        numpy
        scipy
        matplotlib

    Inputs:
        X (array-like): X-coordinates of data points.
        Y (array-like): Y-coordinates of data points.
        Z (array-like): Values at data points.

        variogram_model (string, optional): Specified which variogram model to use;
            may be one of the following: linear, power, gaussian, spherical,
            exponential. Default is linear variogram model. To utilize as custom variogram
            model, specify 'custom'; you must also provide variogram_parameters and
            variogram_function.
        variogram_parameters (array-like, optional): Parameters that define the
            specified variogram model. If not provided, parameters will be automatically
            calculated such that the root-mean-square error for the fit variogram
            function is minimized.
                linear - [slope, nugget]
                power - [scale, exponent, nugget]
                gaussian - [sill, range, nugget]
                spherical - [sill, range, nugget]
                exponential - [sill, range, nugget]
            For a custom variogram model, the parameters are required, as custom variogram
            models currently will not automatically be fit to the data. The code does not
            check that the provided list contains the appropriate number of parameters for
            the custom variogram model, so an incorrect parameter list in such a case will
            probably trigger an esoteric exception someplace deep in the code.
        variogram_function (callable, optional): A callable function that must be provided
            if variogram_model is specified as 'custom'. The function must take only two
            arguments: first, a list of parameters for the variogram model; second, the
            distances at which to calculate the variogram model. The list provided in
            variogram_parameters will be passed to the function as the first argument.
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
            Default is 0 (no rotation). Note that the coordinate system is rotated.

        drift_terms (list of strings, optional): List of drift terms to include in
            universal kriging. Supported drift terms are currently
            'regional_linear', 'point_log', and 'external_Z'.
        point_drift (array-like, optional): Array-like object that contains the
            coordinates and strengths of the point-logarithmic drift terms. Array
            shape must be Nx3, where N is the number of point drift terms. First
            column (index 0) must contain x-coordinates, second column (index 1)
            must contain y-coordinates, and third column (index 2) must contain
            the strengths of each point term. Strengths are relative, so only
            the relation of the values to each other matters. Note that the code
            will appropriately deal with point-logarithmic terms that are at the
            same coordinates as an evaluation point or data point, but Python will
            still kick out a warning message that an ln(0) has been encountered.
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
                variogram_function (callable, optional): A callable function that must be
                    provided if variogram_model is specified as 'custom'. See above for
                    more information.
                nlags (int, optional): Number of averaging bins for the semivariogram.
                    Defualt is 6.
                weight (boolean, optional): Flag that specifies if semivariance at smaller lags
                    should be weighted more heavily when automatically calculating variogram model.
                    True indicates that weights will be applied. Default is False.
                anisotropy_scaling (float, optional): Scalar stretching value to
                    take into account anisotropy. Default is 1 (effectively no
                    stretching). Scaling is applied in the y-direction.
                anisotropy_angle (float, optional): CCW angle (in degrees) by which to
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
                backend (string, optional): Specifies which approach to use in kriging.
                    Specifying 'vectorized' will solve the entire kriging problem at once in a
                    vectorized operation. This approach is faster but also can consume a
                    significant amount of memory for large grids and/or large datasets.
                    Specifying 'loop' will loop through each point at which the kriging system
                    is to be solved. This approach is slower but also less memory-intensive.
                    Default is 'vectorized'.
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

    UNBIAS = True   # This can be changed to remove the unbiasedness condition
                    # Really for testing purposes only...
    eps = 1.e-10    # Cutoff for comparison to zero
    variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model}

    def __init__(self, x, y, z, variogram_model='linear', variogram_parameters=None,
                 variogram_function=None, nlags=6, weight=False, anisotropy_scaling=1.0,
                 anisotropy_angle=0.0, drift_terms=None, point_drift=None,
                 external_drift=None, external_drift_x=None, external_drift_y=None,
                 verbose=False, enable_plotting=False):

        # Deal with mutable default argument
        if drift_terms is None:
            drift_terms = []

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
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]
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
            elif self.variogram_model == 'custom':
                print "Using Custom Variogram Model"
            else:
                print "Using '%s' Variogram Model" % self.variogram_model
                print "Sill:", self.variogram_model_parameters[0]
                print "Range:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2]
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = core.find_statistics(self.X_ADJUSTED, self.Y_ADJUSTED,
                                                                    self.Z, self.variogram_function,
                                                                    self.variogram_model_parameters)
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
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
            if external_drift.shape[0] != external_drift_y.shape[0] or \
               external_drift.shape[1] != external_drift_x.shape[0]:
                if external_drift.shape[0] == external_drift_x.shape[0] and \
                   external_drift.shape[1] == external_drift_y.shape[0]:
                    self.external_Z_drift = np.array(external_drift.T)
                else:
                    raise ValueError("External drift dimensions do not match provided "
                                     "x- and y-coordinate dimensions.")
            else:
                self.external_Z_array = np.array(external_drift)
            self.external_Z_array_x = np.array(external_drift_x).flatten()
            self.external_Z_array_y = np.array(external_drift_y).flatten()
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

    def _calculate_data_point_zscalars(self, x, y, type_='array'):
        """Determines the Z-scalar values at the specified coordinates
        for use when setting up the kriging matrix. Uses bilinear
        interpolation.
        Currently, the Z scalar values are extracted from the input Z grid
        exactly at the specified coordinates. This means that if the Z grid
        resolution is finer than the resolution of the desired kriged grid,
        there is no averaging of the scalar values to return an average
        Z value for that cell in the kriged grid. Rather, the exact Z value
        right at the coordinate is used."""

        if type_ == 'scalar':
            nx = 1
            ny = 1
            z_scalars = None
        else:
            if x.ndim == 1:
                nx = x.shape[0]
                ny = 1
            else:
                ny = x.shape[0]
                nx = x.shape[1]
            z_scalars = np.zeros(x.shape)

        for m in range(ny):
            for n in range(nx):

                if type_ == 'scalar':
                    xn = x
                    yn = y
                else:
                    if x.ndim == 1:
                        xn = x[n]
                        yn = y[n]
                    else:
                        xn = x[m, n]
                        yn = y[m, n]

                if xn > np.amax(self.external_Z_array_x) or xn < np.amin(self.external_Z_array_x) or \
                   yn > np.amax(self.external_Z_array_y) or yn < np.amin(self.external_Z_array_y):
                    raise ValueError("External drift array does not cover specified kriging domain.")

                # bilinear interpolation
                external_x2_index = np.amin(np.where(self.external_Z_array_x >= xn)[0])
                external_x1_index = np.amax(np.where(self.external_Z_array_x <= xn)[0])
                external_y2_index = np.amin(np.where(self.external_Z_array_y >= yn)[0])
                external_y1_index = np.amax(np.where(self.external_Z_array_y <= yn)[0])
                if external_y1_index == external_y2_index:
                    if external_x1_index == external_x2_index:
                        z = self.external_Z_array[external_y1_index, external_x1_index]
                    else:
                        z = (self.external_Z_array[external_y1_index, external_x1_index] *
                             (self.external_Z_array_x[external_x2_index] - xn) +
                             self.external_Z_array[external_y2_index, external_x2_index] *
                             (xn - self.external_Z_array_x[external_x1_index])) / \
                            (self.external_Z_array_x[external_x2_index] -
                             self.external_Z_array_x[external_x1_index])
                elif external_x1_index == external_x2_index:
                    if external_y1_index == external_y2_index:
                        z = self.external_Z_array[external_y1_index, external_x1_index]
                    else:
                        z = (self.external_Z_array[external_y1_index, external_x1_index] *
                             (self.external_Z_array_y[external_y2_index] - yn) +
                             self.external_Z_array[external_y2_index, external_x2_index] *
                             (yn - self.external_Z_array_y[external_y1_index])) / \
                            (self.external_Z_array_y[external_y2_index] -
                             self.external_Z_array_y[external_y1_index])
                else:
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

                if type_ == 'scalar':
                    z_scalars = z
                else:
                    if z_scalars.ndim == 1:
                        z_scalars[n] = z
                    else:
                        z_scalars[m, n] = z

        return z_scalars

    def update_variogram_model(self, variogram_model, variogram_parameters=None,
                               variogram_function=None, nlags=6, weight=False,
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
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]
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
            elif self.variogram_model == 'custom':
                print "Using Custom Variogram Model"
            else:
                print "Using '%s' Variogram Model" % self.variogram_model
                print "Sill:", self.variogram_model_parameters[0]
                print "Range:", self.variogram_model_parameters[1]
                print "Nugget:", self.variogram_model_parameters[2]
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print "Calculating statistics on variogram model fit..."
        self.delta, self.sigma, self.epsilon = core.find_statistics(self.X_ADJUSTED, self.Y_ADJUSTED,
                                                                    self.Z, self.variogram_function,
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
        """Returns the epsilon residuals for the variogram fit."""
        return self.epsilon

    def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
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

    def _get_kriging_matrix(self, n, n_withdrifts):
        """Assembles the kriging matrix."""

        xy = np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1)
        d = cdist(xy, xy, 'euclidean')
        if self.UNBIAS:
            a = np.zeros((n_withdrifts+1, n_withdrifts+1))
        else:
            a = np.zeros((n_withdrifts, n_withdrifts))
        a[:n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
        np.fill_diagonal(a, 0.)

        i = n
        if self.regional_linear_drift:
            a[:n, i] = self.X_ADJUSTED
            a[i, :n] = self.X_ADJUSTED
            i += 1
            a[:n, i] = self.Y_ADJUSTED
            a[i, :n] = self.Y_ADJUSTED
            i += 1
        if self.point_log_drift:
            for well_no in range(self.point_log_array.shape[0]):
                log_dist = np.log(np.sqrt((self.X_ADJUSTED - self.point_log_array[well_no, 0])**2 +
                                          (self.Y_ADJUSTED - self.point_log_array[well_no, 1])**2))
                if np.any(np.isinf(log_dist)):
                    log_dist[np.isinf(log_dist)] = -100.0
                a[:n, i] = - self.point_log_array[well_no, 2] * log_dist
                a[i, :n] = - self.point_log_array[well_no, 2] * log_dist
                i += 1
        if self.external_Z_drift:
            a[:n, i] = self.z_scalars
            a[i, :n] = self.z_scalars
            i += 1
        if i != n_withdrifts:
            print "WARNING: Error in creating kriging matrix. Kriging may fail."
        if self.UNBIAS:
            a[n_withdrifts, :n] = 1.0
            a[:n, n_withdrifts] = 1.0
            a[n:n_withdrifts + 1, n:n_withdrifts + 1] = 0.0

        return a

    def _exec_vector(self, style, xpoints, ypoints, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        nx = xpoints.shape[0]
        ny = ypoints.shape[0]
        n = self.X_ADJUSTED.shape[0]
        n_withdrifts = n
        if self.regional_linear_drift:
            n_withdrifts += 2
        if self.point_log_drift:
            n_withdrifts += self.point_log_array.shape[0]
        if self.external_Z_drift:
            n_withdrifts += 1
        zero_index = None
        zero_value = False
        a_inv = scipy.linalg.inv(self._get_kriging_matrix(n, n_withdrifts))

        if style == 'grid':

            grid_x, grid_y = np.meshgrid(xpoints, ypoints)
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y, self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            bd = cdist(np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=2).reshape((nx*ny, 2)),
                       np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1),
                       'euclidean').reshape((ny, nx, n))
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            if self.UNBIAS:
                b = np.zeros((ny, nx, n_withdrifts+1, 1))
            else:
                b = np.zeros((ny, nx, n_withdrifts, 1))
            b[:, :, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], zero_index[2], 0] = 0.0

            i = n
            if self.regional_linear_drift:
                b[:, :, i, 0] = grid_x[:, :]
                i += 1
                b[:, :, i, 0] = grid_y[:, :]
                i += 1
            if self.point_log_drift:
                for well_no in range(self.point_log_array.shape[0]):
                    log_dist = np.log(np.sqrt((grid_x[:, :] - self.point_log_array[well_no, 0])**2 +
                                              (grid_y[:, :] - self.point_log_array[well_no, 1])**2))
                    if np.any(np.isinf(log_dist)):
                        log_dist[np.isinf(log_dist)] = -100.0
                    b[:, :, i, 0] = - self.point_log_array[well_no, 2] * log_dist
                    i += 1
            if self.external_Z_drift:
                b[:, :, i, 0] = self._calculate_data_point_zscalars(grid_x[:, :], grid_y[:, :])
                i += 1
            if i != n_withdrifts:
                print "WARNING: Error in setting up kriging system. Kriging may fail."
            if self.UNBIAS:
                b[:, :, n_withdrifts, 0] = 1.0

            if self.UNBIAS:
                x = np.dot(a_inv, b.reshape((nx*ny, n_withdrifts+1)).T).reshape((1, n_withdrifts+1, ny, nx)).\
                    T.swapaxes(0, 1)
            else:
                x = np.dot(a_inv, b.reshape((nx*ny, n_withdrifts)).T).reshape((1, n_withdrifts, ny, nx)).\
                    T.swapaxes(0, 1)
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
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y, self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            bd = cdist(np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=2).reshape((nx*ny, 2)),
                       np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1),
                       'euclidean').reshape((ny, nx, n))
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            if self.UNBIAS:
                b = np.zeros((ny, nx, n_withdrifts+1, 1))
            else:
                b = np.zeros((ny, nx, n_withdrifts, 1))
            b[:, :, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], zero_index[2], 0] = 0.0

            i = n
            if self.regional_linear_drift:
                b[:, :, i, 0] = grid_x[:, :]
                i += 1
                b[:, :, i, 0] = grid_y[:, :]
                i += 1
            if self.point_log_drift:
                for well_no in range(self.point_log_array.shape[0]):
                    log_dist = np.log(np.sqrt((grid_x[:, :] - self.point_log_array[well_no, 0])**2 +
                                              (grid_y[:, :] - self.point_log_array[well_no, 1])**2))
                    if np.any(np.isinf(log_dist)):
                        log_dist[np.isinf(log_dist)] = -100.0
                    b[:, :, i, 0] = - self.point_log_array[well_no, 2] * log_dist
                    i += 1
            if self.external_Z_drift:
                b[:, :, i, 0] = self._calculate_data_point_zscalars(grid_x[:, :], grid_y[:, :])
                i += 1
            if i != n_withdrifts:
                print "WARNING: Error in setting up kriging system. Kriging may fail."
            if self.UNBIAS:
                b[:, :, n_withdrifts, 0] = 1.0

            mask_b = np.repeat(mask[:, :, np.newaxis, np.newaxis], n_withdrifts+1, axis=2)
            b = np.ma.array(b, mask=mask_b)
            if self.UNBIAS:
                x = np.dot(a_inv, b.reshape((nx*ny, n_withdrifts+1)).T).reshape((1, n_withdrifts+1, ny, nx)).\
                    T.swapaxes(0, 1)
            else:
                x = np.dot(a_inv, b.reshape((nx*ny, n_withdrifts)).T).reshape((1, n_withdrifts, ny, nx)).\
                    T.swapaxes(0, 1)
            zvalues = np.sum(x[:, :, :n, 0] * self.Z, axis=2)
            sigmasq = np.sum(x[:, :, :, 0] * -b[:, :, :, 0], axis=2)

        elif style == 'points':
            if xpoints.shape != ypoints.shape:
                raise ValueError("xpoints and ypoints must have same dimensions "
                                 "when treated as listing discrete points.")

            xpoints, ypoints = core.adjust_for_anisotropy(xpoints, ypoints, self.XCENTER, self.YCENTER,
                                                          self.anisotropy_scaling, self.anisotropy_angle)

            bd = cdist(np.concatenate((xpoints[:, np.newaxis], ypoints[:, np.newaxis]), axis=1),
                       np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1),
                       'euclidean')
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            if self.UNBIAS:
                b = np.zeros((nx, n_withdrifts+1, 1))
            else:
                b = np.zeros((nx, n_withdrifts, 1))
            b[:, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], zero_index[1], 0] = 0.0

            i = n
            if self.regional_linear_drift:
                b[:, i, 0] = xpoints
                i += 1
                b[:, i, 0] = ypoints
                i += 1
            if self.point_log_drift:
                for well_no in range(self.point_log_array.shape[0]):
                    log_dist = np.log(np.sqrt((xpoints - self.point_log_array[well_no, 0])**2 +
                                              (ypoints - self.point_log_array[well_no, 1])**2))
                    if np.any(np.isinf(log_dist)):
                        log_dist[np.isinf(log_dist)] = -100.0
                    b[:, i, 0] = - self.point_log_array[well_no, 2] * log_dist
                    i += 1
            if self.external_Z_drift:
                b[:, i, 0] = self._calculate_data_point_zscalars(xpoints, ypoints)
            if self.UNBIAS:
                b[:, n_withdrifts, 0] = 1.0

            if self.UNBIAS:
                x = np.dot(a_inv, b.reshape((nx, n_withdrifts+1)).T).reshape((1, n_withdrifts+1, nx)).T
            else:
                x = np.dot(a_inv, b.reshape((nx, n_withdrifts)).T).reshape((1, n_withdrifts, nx)).T
            zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
            sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        return zvalues, sigmasq

    def _exec_loop(self, style, xpoints, ypoints, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        nx = xpoints.shape[0]
        ny = ypoints.shape[0]
        n = self.X_ADJUSTED.shape[0]
        n_withdrifts = n
        if self.regional_linear_drift:
            n_withdrifts += 2
        if self.point_log_drift:
            n_withdrifts += self.point_log_array.shape[0]
        if self.external_Z_drift:
            n_withdrifts += 1
        a_inv = scipy.linalg.inv(self._get_kriging_matrix(n, n_withdrifts))

        if style == 'grid':

            zvalues = np.zeros((ny, nx))
            sigmasq = np.zeros((ny, nx))
            grid_x, grid_y = np.meshgrid(xpoints, ypoints)
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y, self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            for j in range(ny):
                for k in range(nx):
                    bd = np.sqrt((self.X_ADJUSTED - grid_x[j, k])**2 + (self.Y_ADJUSTED - grid_y[j, k])**2)
                    if np.any(np.absolute(bd) <= self.eps):
                        zero_value = True
                        zero_index = np.where(np.absolute(bd) <= self.eps)
                    else:
                        zero_index = None
                        zero_value = False
                    if self.UNBIAS:
                        b = np.zeros((n_withdrifts+1, 1))
                    else:
                        b = np.zeros((n_withdrifts, 1))
                    b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
                    if zero_value:
                        b[zero_index[0], 0] = 0.0

                    i = n
                    if self.regional_linear_drift:
                        b[i, 0] = grid_x[j, k]
                        i += 1
                        b[i, 0] = grid_y[j, k]
                        i += 1
                    if self.point_log_drift:
                        for well_no in range(self.point_log_array.shape[0]):
                            log_dist = np.log(np.sqrt((grid_x[j, k] - self.point_log_array[well_no, 0])**2 +
                                                      (grid_y[j, k] - self.point_log_array[well_no, 1])**2))
                            if np.any(np.isinf(log_dist)):
                                log_dist[np.isinf(log_dist)] = -100.0
                            b[i, 0] = - self.point_log_array[well_no, 2] * log_dist
                            i += 1
                    if self.external_Z_drift:
                        b[i, 0] = self._calculate_data_point_zscalars(grid_x[j, k], grid_y[j, k], type_='scalar')
                        i += 1
                    if i != n_withdrifts:
                        print "WARNING: Error in setting up kriging system. Kriging may fail."
                    if self.UNBIAS:
                        b[n_withdrifts, 0] = 1.0

                    x = np.dot(a_inv, b)
                    zvalues[j, k] = np.sum(x[:n, 0] * self.Z)
                    sigmasq[j, k] = np.sum(x[:, 0] * -b[:, 0])

        elif style == 'masked':
            if mask is None:
                raise IOError("Must specify boolean masking array.")
            if mask.shape[0] != ny or mask.shape[1] != nx:
                if mask.shape[0] == nx and mask.shape[1] == ny:
                    mask = mask.T
                else:
                    raise ValueError("Mask dimensions do not match specified grid dimensions.")

            zvalues = np.zeros((ny, nx))
            sigmasq = np.zeros((ny, nx))
            grid_x, grid_y = np.meshgrid(xpoints, ypoints)
            grid_x, grid_y = core.adjust_for_anisotropy(grid_x, grid_y, self.XCENTER, self.YCENTER,
                                                        self.anisotropy_scaling, self.anisotropy_angle)

            for j in range(ny):
                for k in range(nx):
                    if not mask[j, k]:
                        bd = np.sqrt((self.X_ADJUSTED - grid_x[j, k])**2 + (self.Y_ADJUSTED - grid_y[j, k])**2)
                        if np.any(np.absolute(bd) <= self.eps):
                            zero_value = True
                            zero_index = np.where(np.absolute(bd) <= self.eps)
                        else:
                            zero_index = None
                            zero_value = False
                        if self.UNBIAS:
                            b = np.zeros((n_withdrifts+1, 1))
                        else:
                            b = np.zeros((n_withdrifts, 1))
                        b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
                        if zero_value:
                            b[zero_index[0], 0] = 0.0

                        i = n
                        if self.regional_linear_drift:
                            b[i, 0] = grid_x[j, k]
                            i += 1
                            b[i, 0] = grid_y[j, k]
                            i += 1
                        if self.point_log_drift:
                            for well_no in range(self.point_log_array.shape[0]):
                                log_dist = np.log(np.sqrt((grid_x[j, k] - self.point_log_array[well_no, 0])**2 +
                                                          (grid_y[j, k] - self.point_log_array[well_no, 1])**2))
                                if np.any(np.isinf(log_dist)):
                                    log_dist[np.isinf(log_dist)] = -100.0
                                b[i, 0] = - self.point_log_array[well_no, 2] * log_dist
                                i += 1
                        if self.external_Z_drift:
                            b[i, 0] = self._calculate_data_point_zscalars(grid_x[j, k], grid_y[j, k], type_='scalar')
                            i += 1
                        if i != n_withdrifts:
                            print "WARNING: Error in setting up kriging system. Kriging may fail."
                        if self.UNBIAS:
                            b[n_withdrifts, 0] = 1.0

                        x = np.dot(a_inv, b)
                        zvalues[j, k] = np.sum(x[:n, 0] * self.Z)
                        sigmasq[j, k] = np.sum(x[:, 0] * -b[:, 0])

            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        elif style == 'points':
            if xpoints.shape != ypoints.shape:
                raise ValueError("xpoints and ypoints must have same dimensions "
                                 "when treated as listing discrete points.")

            zvalues = np.zeros(nx)
            sigmasq = np.zeros(nx)
            xpoints, ypoints = core.adjust_for_anisotropy(xpoints, ypoints, self.XCENTER, self.YCENTER,
                                                          self.anisotropy_scaling, self.anisotropy_angle)

            for j in range(nx):
                bd = np.sqrt((self.X_ADJUSTED - xpoints[j])**2 + (self.Y_ADJUSTED - ypoints[j])**2)
                if np.any(np.absolute(bd) <= self.eps):
                    zero_value = True
                    zero_index = np.where(np.absolute(bd) <= self.eps)
                else:
                    zero_index = None
                    zero_value = False
                if self.UNBIAS:
                    b = np.zeros((n_withdrifts+1, 1))
                else:
                    b = np.zeros((n_withdrifts, 1))
                b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
                if zero_value:
                    b[zero_index[0], 0] = 0.0

                i = n
                if self.regional_linear_drift:
                    b[i, 0] = xpoints[j]
                    i += 1
                    b[i, 0] = ypoints[j]
                    i += 1
                if self.point_log_drift:
                    for well_no in range(self.point_log_array.shape[0]):
                        log_dist = np.log(np.sqrt((xpoints[j] - self.point_log_array[well_no, 0])**2 +
                                                  (ypoints[j] - self.point_log_array[well_no, 1])**2))
                        if np.any(np.isinf(log_dist)):
                            log_dist[np.isinf(log_dist)] = -100.0
                        b[i, 0] = - self.point_log_array[well_no, 2] * log_dist
                        i += 1
                if self.external_Z_drift:
                    b[i, 0] = self._calculate_data_point_zscalars(xpoints[j], ypoints[j], type_='scalar')
                    i += 1
                if i != n_withdrifts:
                    print "WARNING: Error in setting up kriging system. Kriging may fail."
                if self.UNBIAS:
                    b[n_withdrifts, 0] = 1.0

                x = np.dot(a_inv, b)
                zvalues[j] = np.sum(x[:n, 0] * self.Z)
                sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        return zvalues, sigmasq

    def execute(self, style, xpoints, ypoints, mask=None, backend='vectorized'):
        """Calculates a kriged grid and the associated variance. Includes drift terms.

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
                False indicates that the point should not be masked, so the kriging system
                will be solved at the point.
                True indicates that the point should be masked, so the kriging system should
                will not be solved at the point.
            backend (string, optional): Specifies which approach to use in kriging.
                Specifying 'vectorized' will solve the entire kriging problem at once in a
                vectorized operation. This approach is faster but also can consume a
                significant amount of memory for large grids and/or large datasets.
                Specifying 'loop' will loop through each point at which the kriging system
                is to be solved. This approach is slower but also less memory-intensive.
                Default is 'vectorized'.
        Outputs:
            zvalues (numpy array, dim MxN or dim Nx1): Z-values of specified grid or at the
                specified set of points. If style was specified as 'masked', zvalues will
                be a numpy masked array.
            sigmasq (numpy array, dim MxN or dim Nx1): Variance at specified grid points or
                at the specified set of points. If style was specified as 'masked', sigmasq
                will be a numpy masked array.
        """

        if self.verbose:
            print "Executing Universal Kriging...\n"

        if style != 'grid' and style != 'masked' and style != 'points':
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpoints = np.array(xpoints, copy=True).flatten()
        ypoints = np.array(ypoints, copy=True).flatten()
        if backend == 'vectorized':
            zvalues, sigmasq = self._exec_vector(style, xpoints, ypoints, mask)
        elif backend == 'loop':
            zvalues, sigmasq = self._exec_loop(style, xpoints, ypoints, mask)
        else:
            raise ValueError('Specified backend {} is not supported.'.format(backend))

        return zvalues, sigmasq
