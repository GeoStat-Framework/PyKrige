# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Contains class OrdinaryKriging3D.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.
.. [2] N. Cressie, Statistics for spatial data,
(Wiley Series in Probability and Statistics, 1993) 137 p.

Copyright (c) 2015-2020, PyKrige Developers
"""

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from . import variogram_models
from . import core
from .core import (
    _adjust_for_anisotropy,
    _initialize_variogram_model,
    _make_variogram_parameter_list,
    _find_statistics,
    P_INV,
)
import warnings


class OrdinaryKriging3D:
    """Three-dimensional ordinary kriging.

    Parameters
    ----------
    x : array_like
        X-coordinates of data points.
    y : array_like
        Y-coordinates of data points.
    z : array_like
        Z-coordinates of data points.
    val : array_like
        Values at data points.
    variogram_model : str or GSTools CovModel, optional
        Specified which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model. To utilize a custom variogram model,
        specify 'custom'; you must also provide variogram_parameters and
        variogram_function. Note that the hole-effect model is only technically
        correct for one-dimensional problems.
        You can also use a
        `GSTools <https://github.com/GeoStat-Framework/GSTools>`_ CovModel.
    variogram_parameters : list or dict, optional
        Parameters that define the specified variogram model. If not provided,
        parameters will be automatically calculated using a "soft" L1 norm
        minimization scheme. For variogram model parameters provided in a dict,
        the required dict keys vary according to the specified variogram
        model: ::

           # linear
               {'slope': slope, 'nugget': nugget}
           # power
               {'scale': scale, 'exponent': exponent, 'nugget': nugget}
           # gaussian, spherical, exponential and hole-effect:
               {'sill': s, 'range': r, 'nugget': n}
               # OR
               {'psill': p, 'range': r, 'nugget': n}

        Note that either the full sill or the partial sill
        (psill = sill - nugget) can be specified in the dict.
        For variogram model parameters provided in a list, the entries
        must be as follows: ::

           # linear
               [slope, nugget]
           # power
               [scale, exponent, nugget]
           # gaussian, spherical, exponential and hole-effect:
               [sill, range, nugget]

        Note that the full sill (NOT the partial sill) must be specified
        in the list format.
        For a custom variogram model, the parameters are required, as custom
        variogram models will not automatically be fit to the data.
        Furthermore, the parameters must be specified in list format, in the
        order in which they are used in the callable function (see
        variogram_function for more information). The code does not check
        that the provided list contains the appropriate number of parameters
        for the custom variogram model, so an incorrect parameter list in
        such a case will probably trigger an esoteric exception someplace
        deep in the code.
        NOTE that, while the list format expects the full sill, the code
        itself works internally with the partial sill.
    variogram_function : callable, optional
        A callable function that must be provided if variogram_model is
        specified as 'custom'. The function must take only two arguments:
        first, a list of parameters for the variogram model;
        second, the distances at which to calculate the variogram model.
        The list provided in variogram_parameters will be passed to the
        function as the first argument.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : boolean, optional
        Flag that specifies if semivariance at smaller lags should be weighted
        more heavily when automatically calculating variogram model.
        The routine is currently hard-coded such  that the weights are
        calculated from a logistic function, so weights at small lags are ~1
        and weights at the longest lags are ~0; the center of the logistic
        weighting is hard-coded to be at 70% of the distance from the shortest
        lag to the largest lag. Setting this parameter to True indicates that
        weights will be applied. Default is False.
        (Kitanidis suggests that the values at smaller lags are more
        important in fitting a variogram model, so the option is provided
        to enable such weighting.)
    anisotropy_scaling_y : float, optional
        Scalar stretching value to take into account anisotropy
        in the y direction. Default is 1 (effectively no stretching).
        Scaling is applied in the y direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle_x/y/z,
        if anisotropy_angle_x/y/z is/are not 0).
    anisotropy_scaling_z : float, optional
        Scalar stretching value to take into account anisotropy
        in the z direction. Default is 1 (effectively no stretching).
        Scaling is applied in the z direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle_x/y/z,
        if anisotropy_angle_x/y/z is/are not 0).
    anisotropy_angle_x : float, optional
        CCW angle (in degrees) by which to rotate coordinate system about
        the x axis in order to take into account anisotropy.
        Default is 0 (no rotation). Note that the coordinate system is rotated.
        X rotation is applied first, then y rotation, then z rotation.
        Scaling is applied after rotation.
    anisotropy_angle_y : float, optional
        CCW angle (in degrees) by which to rotate coordinate system about
        the y axis in order to take into account anisotropy.
        Default is 0 (no rotation). Note that the coordinate system is rotated.
        X rotation is applied first, then y rotation, then z rotation.
        Scaling is applied after rotation.
    anisotropy_angle_z : float, optional
        CCW angle (in degrees) by which to rotate coordinate system about
        the z axis in order to take into account anisotropy.
        Default is 0 (no rotation). Note that the coordinate system is rotated.
        X rotation is applied first, then y rotation, then z rotation.
        Scaling is applied after rotation.
    verbose : bool, optional
        Enables program text output to monitor kriging process.
        Default is False (off).
    enable_plotting : bool, optional
        Enables plotting to display variogram. Default is False (off).
    exact_values : bool, optional
        If True, interpolation provides input values at input locations.
        If False, interpolation accounts for variance/nugget within input
        values at input locations and does not behave as an
        exact-interpolator [2]. Note that this only has an effect if
        there is variance/nugget present within the input data since it is
        interpreted as measurement error. If the nugget is zero, the kriged
        field will behave as an exact interpolator.
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False
    pseudo_inv_type : :class:`str`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `"pinv"`: use `pinv` from `scipy` which uses `lstsq`
            * `"pinv2"`: use `pinv2` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values

        Default: `"pinv"`

    References
    ----------
    .. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
       Hydrogeology, (Cambridge University Press, 1997) 272 p.
    .. [2] N. Cressie, Statistics for spatial data,
       (Wiley Series in Probability and Statistics, 1993) 137 p.
    """

    eps = 1.0e-10  # Cutoff for comparison to zero
    variogram_dict = {
        "linear": variogram_models.linear_variogram_model,
        "power": variogram_models.power_variogram_model,
        "gaussian": variogram_models.gaussian_variogram_model,
        "spherical": variogram_models.spherical_variogram_model,
        "exponential": variogram_models.exponential_variogram_model,
        "hole-effect": variogram_models.hole_effect_variogram_model,
    }

    def __init__(
        self,
        x,
        y,
        z,
        val,
        variogram_model="linear",
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        anisotropy_angle_x=0.0,
        anisotropy_angle_y=0.0,
        anisotropy_angle_z=0.0,
        verbose=False,
        enable_plotting=False,
        exact_values=True,
        pseudo_inv=False,
        pseudo_inv_type="pinv",
    ):
        # config the pseudo inverse
        self.pseudo_inv = bool(pseudo_inv)
        self.pseudo_inv_type = str(pseudo_inv_type)
        if self.pseudo_inv_type not in P_INV:
            raise ValueError("pseudo inv type not valid: " + str(pseudo_inv_type))

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None

        if not isinstance(exact_values, bool):
            raise ValueError("exact_values has to be boolean True or False")
        self.exact_values = exact_values

        # check if a GSTools covariance model is given
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            if self.model.dim < 3:
                raise ValueError("GSTools: model dim is not 3")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling_y = self.model.pykrige_anis_y
            anisotropy_scaling_z = self.model.pykrige_anis_z
            anisotropy_angle_x = self.model.pykrige_angle_x
            anisotropy_angle_y = self.model.pykrige_angle_y
            anisotropy_angle_z = self.model.pykrige_angle_z
        if (
            self.variogram_model not in self.variogram_dict.keys()
            and self.variogram_model != "custom"
        ):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        elif self.variogram_model == "custom":
            if variogram_function is None or not callable(variogram_function):
                raise ValueError(
                    "Must specify callable function for custom variogram model."
                )
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        # Code assumes 1D input arrays. Ensures that any extraneous dimensions
        # don't get in the way. Copies are created to avoid any problems with
        # referencing the original passed arguments.
        self.X_ORIG = np.atleast_1d(
            np.squeeze(np.array(x, copy=True, dtype=np.float64))
        )
        self.Y_ORIG = np.atleast_1d(
            np.squeeze(np.array(y, copy=True, dtype=np.float64))
        )
        self.Z_ORIG = np.atleast_1d(
            np.squeeze(np.array(z, copy=True, dtype=np.float64))
        )
        self.VALUES = np.atleast_1d(
            np.squeeze(np.array(val, copy=True, dtype=np.float64))
        )

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG)) / 2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG)) / 2.0
        self.ZCENTER = (np.amax(self.Z_ORIG) + np.amin(self.Z_ORIG)) / 2.0
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.anisotropy_angle_x = anisotropy_angle_x
        self.anisotropy_angle_y = anisotropy_angle_y
        self.anisotropy_angle_z = anisotropy_angle_z
        if self.verbose:
            print("Adjusting data for anisotropy...")
        self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED = _adjust_for_anisotropy(
            np.vstack((self.X_ORIG, self.Y_ORIG, self.Z_ORIG)).T,
            [self.XCENTER, self.YCENTER, self.ZCENTER],
            [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
            [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z],
        ).T

        if self.verbose:
            print("Initializing variogram model...")

        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
            self.VALUES,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            "euclidean",
        )

        if self.verbose:
            if self.variogram_model == "linear":
                print("Using '%s' Variogram Model" % "linear")
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], "\n")
            elif self.variogram_model == "power":
                print("Using '%s' Variogram Model" % "power")
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
            elif self.variogram_model == "custom":
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print(
                    "Full Sill:",
                    self.variogram_model_parameters[0]
                    + self.variogram_model_parameters[2],
                )
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = _find_statistics(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
            self.VALUES,
            self.variogram_function,
            self.variogram_model_parameters,
            "euclidean",
            self.pseudo_inv,
        )
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, "\n")

    def update_variogram_model(
        self,
        variogram_model,
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        anisotropy_angle_x=0.0,
        anisotropy_angle_y=0.0,
        anisotropy_angle_z=0.0,
    ):
        """Changes the variogram model and variogram parameters for
        the kriging system.

        Parameters
        ----------
        variogram_model : str or GSTools CovModel
            May be any of the variogram models listed above.
            May also be 'custom', in which case variogram_parameters and
            variogram_function must be specified.
            You can also use a
            `GSTools <https://github.com/GeoStat-Framework/GSTools>`_ CovModel.
        variogram_parameters : list or dict, optional
            List or dict of variogram model parameters, as explained above.
            If not provided, a best fit model will be calculated as
            described above.
        variogram_function : callable, optional
            A callable function that must be provided if variogram_model is
            specified as 'custom'. See above for more information.
        nlags : int, optional
            Number of averaging bins for the semivariogram. Default is 6.
        weight : bool, optional
            Flag that specifies if semivariance at smaller lags should be
            weighted more heavily when automatically calculating
            variogram model. See above for more information. True indicates
            that weights will be applied. Default is False.
        anisotropy_scaling_y : float, optional
            Scalar stretching value to take into account anisotropy
            in y-direction. Default is 1 (effectively no stretching).
            See above for more information.
        anisotropy_scaling_z : float, optional
            Scalar stretching value to take into account anisotropy
            in z-direction. Default is 1 (effectively no stretching).
            See above for more information.
        anisotropy_angle_x : float, optional
            Angle (in degrees) by which to rotate coordinate system about
            the x axis in order to take into account anisotropy.
            Default is 0 (no rotation). See above for more information.
        anisotropy_angle_y : float, optional
            Angle (in degrees) by which to rotate coordinate system about
            the y axis in order to take into account anisotropy.
            Default is 0 (no rotation). See above for more information.
        anisotropy_angle_z : float, optional
            Angle (in degrees) by which to rotate coordinate system about
            the z axis in order to take into account anisotropy.
            Default is 0 (no rotation). See above for more information.
        """

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None
        # check if a GSTools covariance model is given
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            if self.model.dim < 3:
                raise ValueError("GSTools: model dim is not 3")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling_y = self.model.pykrige_anis_y
            anisotropy_scaling_z = self.model.pykrige_anis_z
            anisotropy_angle_x = self.model.pykrige_angle_x
            anisotropy_angle_y = self.model.pykrige_angle_y
            anisotropy_angle_z = self.model.pykrige_angle_z
        if (
            self.variogram_model not in self.variogram_dict.keys()
            and self.variogram_model != "custom"
        ):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        elif self.variogram_model == "custom":
            if variogram_function is None or not callable(variogram_function):
                raise ValueError(
                    "Must specify callable function for custom variogram model."
                )
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        if (
            anisotropy_scaling_y != self.anisotropy_scaling_y
            or anisotropy_scaling_z != self.anisotropy_scaling_z
            or anisotropy_angle_x != self.anisotropy_angle_x
            or anisotropy_angle_y != self.anisotropy_angle_y
            or anisotropy_angle_z != self.anisotropy_angle_z
        ):
            if self.verbose:
                print("Adjusting data for anisotropy...")
            self.anisotropy_scaling_y = anisotropy_scaling_y
            self.anisotropy_scaling_z = anisotropy_scaling_z
            self.anisotropy_angle_x = anisotropy_angle_x
            self.anisotropy_angle_y = anisotropy_angle_y
            self.anisotropy_angle_z = anisotropy_angle_z
            self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED = _adjust_for_anisotropy(
                np.vstack((self.X_ORIG, self.Y_ORIG, self.Z_ORIG)).T,
                [self.XCENTER, self.YCENTER, self.ZCENTER],
                [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
                [
                    self.anisotropy_angle_x,
                    self.anisotropy_angle_y,
                    self.anisotropy_angle_z,
                ],
            ).T

        if self.verbose:
            print("Updating variogram mode...")

        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
            self.VALUES,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            "euclidean",
        )

        if self.verbose:
            if self.variogram_model == "linear":
                print("Using '%s' Variogram Model" % "linear")
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], "\n")
            elif self.variogram_model == "power":
                print("Using '%s' Variogram Model" % "power")
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
            elif self.variogram_model == "custom":
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print(
                    "Full Sill:",
                    self.variogram_model_parameters[0]
                    + self.variogram_model_parameters[2],
                )
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = _find_statistics(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
            self.VALUES,
            self.variogram_function,
            self.variogram_model_parameters,
            "euclidean",
            self.pseudo_inv,
        )
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, "\n")

    def display_variogram_model(self):
        """Displays variogram model with the actual binned data."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lags, self.semivariance, "r*")
        ax.plot(
            self.lags,
            self.variogram_function(self.variogram_model_parameters, self.lags),
            "k-",
        )
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
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c="k", marker="*")
        ax.axhline(y=0.0)
        plt.show()

    def get_statistics(self):
        """Returns the Q1, Q2, and cR statistics for the
        variogram fit (in that order). No arguments.
        """
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        """Prints out the Q1, Q2, and cR statistics for the variogram fit.
        NOTE that ideally Q1 is close to zero, Q2 is close to 1,
        and cR is as small as possible.
        """
        print("Q1 =", self.Q1)
        print("Q2 =", self.Q2)
        print("cR =", self.cR)

    def _get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""

        xyz = np.concatenate(
            (
                self.X_ADJUSTED[:, np.newaxis],
                self.Y_ADJUSTED[:, np.newaxis],
                self.Z_ADJUSTED[:, np.newaxis],
            ),
            axis=1,
        )
        d = cdist(xyz, xyz, "euclidean")
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = -self.variogram_function(self.variogram_model_parameters, d)
        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv = P_INV[self.pseudo_inv_type](a)
        else:
            a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n + 1, 1))
        b[:, :n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value and self.exact_values:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n + 1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
        kvalues = np.sum(x[:, :n, 0] * self.VALUES, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return kvalues, sigmasq

    def _exec_loop(self, a, bd_all, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        kvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv = P_INV[self.pseudo_inv_type](a)
        else:
            a_inv = scipy.linalg.inv(a)

        for j in np.nonzero(~mask)[
            0
        ]:  # Note that this is the same thing as range(npt) if mask is not defined,
            bd = bd_all[j]  # otherwise it takes the non-masked elements.
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_value = False
                zero_index = None

            b = np.zeros((n + 1, 1))
            b[:n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value and self.exact_values:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            x = np.dot(a_inv, b)
            kvalues[j] = np.sum(x[:n, 0] * self.VALUES)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        return kvalues, sigmasq

    def _exec_loop_moving_window(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Uses only a certain number of closest points. Not very memory intensive,
        but the loop is done in pure Python.
        """
        import scipy.linalg.lapack

        npt = bd_all.shape[0]
        n = bd_idx.shape[1]
        kvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        for i in np.nonzero(~mask)[0]:
            b_selector = bd_idx[i]
            bd = bd_all[i]

            a_selector = np.concatenate((b_selector, np.array([a_all.shape[0] - 1])))
            a = a_all[a_selector[:, None], a_selector]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_value = False
                zero_index = None
            b = np.zeros((n + 1, 1))
            b[:n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value and self.exact_values:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            x = scipy.linalg.solve(a, b)

            kvalues[i] = x[:n, 0].dot(self.VALUES[b_selector])
            sigmasq[i] = -x[:, 0].dot(b[:, 0])

        return kvalues, sigmasq

    def execute(
        self,
        style,
        xpoints,
        ypoints,
        zpoints,
        mask=None,
        backend="vectorized",
        n_closest_points=None,
    ):
        """Calculates a kriged grid and the associated variance.

        This is now the method that performs the main kriging calculation.
        Note that currently measurements (i.e., z values) are
        considered 'exact'. This means that, when a specified coordinate
        for interpolation is exactly the same as one of the data points,
        the variogram evaluated at the point is forced to be zero.
        Also, the diagonal of the kriging matrix is also always forced
        to be zero. In forcing the variogram evaluated at data points
        to be zero, we are effectively saying that there is no variance
        at that point (no uncertainty, so the value is 'exact').

        In the future, the code may include an extra 'exact_values' boolean
        flag that can be adjusted to specify whether to treat the
        measurements as 'exact'. Setting the flag to false would indicate
        that the variogram should not be forced to be zero at zero distance
        (i.e., when evaluated at data points). Instead, the uncertainty in the
        point will be equal to the nugget. This would mean that the diagonal
        of the kriging matrix would be set to the nugget instead of to zero.

        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points.
            Specifying 'grid' treats xpoints, ypoints, and zpoints as arrays of
            x, y, and z coordinates that define a rectangular grid.
            Specifying 'points' treats xpoints, ypoints, and zpoints as arrays
            that provide coordinates at which to solve the kriging system.
            Specifying 'masked' treats xpoints, ypoints, and zpoints as arrays
            of x, y, and z coordinates that define a rectangular grid and uses
            mask to only evaluate specific points in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked', x-coordinates of
            LxMxN grid. If style is specified as 'points', x-coordinates of
            specific points at which to solve kriging system.
        ypoints : array-like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked', y-coordinates of
            LxMxN grid. If style is specified as 'points', y-coordinates of
            specific points at which to solve kriging system.
            Note that in this case, xpoints, ypoints, and zpoints must have the
            same dimensions (i.e., L = M = N).
        zpoints : array-like, shape (L,) or (L, 1)
            If style is specified as 'grid' or 'masked', z-coordinates of
            LxMxN grid. If style is specified as 'points', z-coordinates of
            specific points at which to solve kriging system.
            Note that in this case, xpoints, ypoints, and zpoints must have the
            same dimensions (i.e., L = M = N).
        mask : boolean array, shape (L, M, N), optional
            Specifies the points in the rectangular grid defined by xpoints,
            ypoints, zpoints that are to be excluded in the
            kriging calculations. Must be provided if style is specified
            as 'masked'. False indicates that the point should not be masked,
            so the kriging system will be solved at the point.
            True indicates that the point should be masked, so the kriging
            system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging. Specifying 'vectorized'
            will solve the entire kriging problem at once in a
            vectorized operation. This approach is faster but also can consume a
            significant amount of memory for large grids and/or large datasets.
            Specifying 'loop' will loop through each point at which the kriging
            system is to be solved. This approach is slower but also less
            memory-intensive. Default is 'vectorized'.
        n_closest_points : int, optional
            For kriging with a moving window, specifies the number of nearby
            points to use in the calculation. This can speed up the calculation
            for large datasets, but should be used with caution.
            As Kitanidis notes, kriging with a moving window can produce
            unexpected oddities if the variogram model is not carefully chosen.

        Returns
        -------
        kvalues : ndarray, shape (L, M, N) or (N, 1)
            Interpolated values of specified grid or at the specified set
            of points. If style was specified as 'masked', kvalues will be a
            numpy masked array.
        sigmasq : ndarray, shape (L, M, N) or (N, 1)
            Variance at specified grid points or at the specified set of points.
            If style was specified as 'masked', sigmasq will be a numpy
            masked array.
        """

        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        if style != "grid" and style != "masked" and style != "points":
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        zpts = np.atleast_1d(np.squeeze(np.array(zpoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        nz = zpts.size
        a = self._get_kriging_matrix(n)

        if style in ["grid", "masked"]:
            if style == "masked":
                if mask is None:
                    raise IOError(
                        "Must specify boolean masking array when style is 'masked'."
                    )
                if mask.ndim != 3:
                    raise ValueError("Mask is not three-dimensional.")
                if mask.shape[0] != nz or mask.shape[1] != ny or mask.shape[2] != nx:
                    if (
                        mask.shape[0] == nx
                        and mask.shape[2] == nz
                        and mask.shape[1] == ny
                    ):
                        mask = mask.swapaxes(0, 2)
                    else:
                        raise ValueError(
                            "Mask dimensions do not match specified grid dimensions."
                        )
                mask = mask.flatten()
            npt = nz * ny * nx
            grid_z, grid_y, grid_x = np.meshgrid(zpts, ypts, xpts, indexing="ij")
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()
            zpts = grid_z.flatten()
        elif style == "points":
            if xpts.size != ypts.size and ypts.size != zpts.size:
                raise ValueError(
                    "xpoints, ypoints, and zpoints must have "
                    "same dimensions when treated as listing "
                    "discrete points."
                )
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts, ypts, zpts = _adjust_for_anisotropy(
            np.vstack((xpts, ypts, zpts)).T,
            [self.XCENTER, self.YCENTER, self.ZCENTER],
            [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
            [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z],
        ).T

        if style != "masked":
            mask = np.zeros(npt, dtype="bool")

        xyz_points = np.concatenate(
            (zpts[:, np.newaxis], ypts[:, np.newaxis], xpts[:, np.newaxis]), axis=1
        )
        xyz_data = np.concatenate(
            (
                self.Z_ADJUSTED[:, np.newaxis],
                self.Y_ADJUSTED[:, np.newaxis],
                self.X_ADJUSTED[:, np.newaxis],
            ),
            axis=1,
        )
        bd = cdist(xyz_points, xyz_data, "euclidean")

        if n_closest_points is not None:
            from scipy.spatial import cKDTree

            tree = cKDTree(xyz_data)
            bd, bd_idx = tree.query(xyz_points, k=n_closest_points, eps=0.0)
            if backend == "loop":
                kvalues, sigmasq = self._exec_loop_moving_window(a, bd, mask, bd_idx)
            else:
                raise ValueError(
                    "Specified backend '{}' not supported "
                    "for moving window.".format(backend)
                )
        else:
            if backend == "vectorized":
                kvalues, sigmasq = self._exec_vector(a, bd, mask)
            elif backend == "loop":
                kvalues, sigmasq = self._exec_loop(a, bd, mask)
            else:
                raise ValueError(
                    "Specified backend {} is not supported for "
                    "3D ordinary kriging.".format(backend)
                )

        if style == "masked":
            kvalues = np.ma.array(kvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ["masked", "grid"]:
            kvalues = kvalues.reshape((nz, ny, nx))
            sigmasq = sigmasq.reshape((nz, ny, nx))

        return kvalues, sigmasq
