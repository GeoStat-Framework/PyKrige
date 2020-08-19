# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Contains class UniversalKriging, provides greater control over 2D kriging by
utilizing drift terms.

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


class UniversalKriging:
    """Provides greater control over 2D kriging by utilizing drift terms.

    Parameters
    ----------
    x : array_like
        X-coordinates of data points.
    y : array_like
        Y-coordinates of data points.
    z : array_like
        Values at data points.
    variogram_model: str or GSTools CovModel, optional
        Specified which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model. To utilize a custom variogram model,
        specify 'custom'; you must also provide variogram_parameters and
        variogram_function. Note that the hole-effect model is only
        technically correct for one-dimensional problems.
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
        first, a list of parameters for the variogram model; second,
        the distances at which to calculate the variogram model. The list
        provided in variogram_parameters will be passed to the function
        as the first argument.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : bool, optional
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
    anisotropy_scaling : float, optional
        Scalar stretching value to take into account anisotropy.
        Default is 1 (effectively no stretching).
        Scaling is applied in the y-direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
        is not 0).
    anisotropy_angle : float, optional
        CCW angle (in degrees) by which to rotate coordinate system in order
        to take into account anisotropy. Default is 0 (no rotation).
        Note that the coordinate system is rotated.
    drift_terms : list of strings, optional
        List of drift terms to include in universal kriging. Supported drift
        terms are currently 'regional_linear', 'point_log', 'external_Z',
        'specified', and 'functional'.
    point_drift : array_like, optional
        Array-like object that contains the coordinates and strengths of the
        point-logarithmic drift terms. Array shape must be (N, 3), where N is
        the number of point drift terms. First column (index 0) must contain
        x-coordinates, second column (index 1) must contain y-coordinates,
        and third column (index 2) must contain the strengths of each
        point term. Strengths are relative, so only the relation of the values
        to each other matters. Note that the code will appropriately deal with
        point-logarithmic terms that are at the same coordinates as an
        evaluation point or data point, but Python will still kick out a
        warning message that an ln(0) has been encountered. If the problem
        involves anisotropy, the well coordinates will be adjusted and the
        drift values will be calculated in the adjusted data frame.
    external_drift : array_like, optional
        Gridded data used for the external Z scalar drift term.
        Must be shape (M, N), where M is in the y-direction and N is in the
        x-direction. Grid spacing does not need to be constant. If grid spacing
        is not constant, must specify the grid cell sizes. If the problem
        involves anisotropy, the external drift values are extracted based on
        the pre-adjusted coordinates (i.e., the original coordinate system).
    external_drift_x : array_like, optional
        X-coordinates for gridded external Z-scalar data. Must be shape (M,)
        or (M, 1), where M is the number of grid cells in the x-direction.
        The coordinate is treated as the center of the cell.
    external_drift_y : array_like, optional
        Y-coordinates for gridded external Z-scalar data. Must be shape (N,)
        or (N, 1), where N is the number of grid cells in the y-direction.
        The coordinate is treated as the center of the cell.
    specified_drift : list of array-like objects, optional
        List of arrays that contain the drift values at data points.
        The arrays must be shape (N,) or (N, 1), where N is the number of
        data points. Any number of specified-drift terms may be used.
    functional_drift : list of callable objects, optional
        List of callable functions that will be used to evaluate drift terms.
        The function must be a function of only the two spatial coordinates
        and must return a single value for each coordinate pair.
        It must be set up to be called with only two arguments, first an array
        of x values and second an array of y values. If the problem involves
        anisotropy, the drift values are calculated in the adjusted data frame.
    verbose : bool, optional
        Enables program text output to monitor kriging process.
        Default is False (off).
    enable_plotting : boolean, optional
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

    UNBIAS = True  # This can be changed to remove the unbiasedness condition
    # Really for testing purposes only...
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
        variogram_model="linear",
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
        drift_terms=None,
        point_drift=None,
        external_drift=None,
        external_drift_x=None,
        external_drift_y=None,
        specified_drift=None,
        functional_drift=None,
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

        # Deal with mutable default argument
        if drift_terms is None:
            drift_terms = []
        if specified_drift is None:
            specified_drift = []
        if functional_drift is None:
            functional_drift = []

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
            if self.model.dim == 3:
                raise ValueError("GSTools: model dim is not 1 or 2")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling = self.model.pykrige_anis
            anisotropy_angle = self.model.pykrige_angle
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
        self.Z = np.atleast_1d(np.squeeze(np.array(z, copy=True, dtype=np.float64)))

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        # adjust for anisotropy...
        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG)) / 2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG)) / 2.0
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        if self.verbose:
            print("Adjusting data for anisotropy...")
        self.X_ADJUSTED, self.Y_ADJUSTED = _adjust_for_anisotropy(
            np.vstack((self.X_ORIG, self.Y_ORIG)).T,
            [self.XCENTER, self.YCENTER],
            [self.anisotropy_scaling],
            [self.anisotropy_angle],
        ).T

        if self.verbose:
            print("Initializing variogram model...")

        # see comment in ok.py about 'use_psill' kwarg...
        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            "euclidean",
        )
        # TODO extend geographic capabilities to UK...

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
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
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

        if self.verbose:
            print("Initializing drift terms...")

        # Note that the regional linear drift values will be based
        # on the adjusted coordinate system, Really, it doesn't actually
        # matter which coordinate system is used here.
        if "regional_linear" in drift_terms:
            self.regional_linear_drift = True
            if self.verbose:
                print("Implementing regional linear drift.")
        else:
            self.regional_linear_drift = False

        # External Z scalars are extracted using the original
        # (unadjusted) coordinates.
        if "external_Z" in drift_terms:
            if external_drift is None:
                raise ValueError("Must specify external Z drift terms.")
            if external_drift_x is None or external_drift_y is None:
                raise ValueError("Must specify coordinates of external Z drift terms.")
            self.external_Z_drift = True
            if (
                external_drift.shape[0] != external_drift_y.shape[0]
                or external_drift.shape[1] != external_drift_x.shape[0]
            ):
                if (
                    external_drift.shape[0] == external_drift_x.shape[0]
                    and external_drift.shape[1] == external_drift_y.shape[0]
                ):
                    self.external_Z_drift = np.array(external_drift.T)
                else:
                    raise ValueError(
                        "External drift dimensions do not match "
                        "provided x- and y-coordinate dimensions."
                    )
            else:
                self.external_Z_array = np.array(external_drift)
            self.external_Z_array_x = np.array(external_drift_x).flatten()
            self.external_Z_array_y = np.array(external_drift_y).flatten()
            self.z_scalars = self._calculate_data_point_zscalars(
                self.X_ORIG, self.Y_ORIG
            )
            if self.verbose:
                print("Implementing external Z drift.")
        else:
            self.external_Z_drift = False

        # Well coordinates are rotated into adjusted coordinate frame.
        if "point_log" in drift_terms:
            if point_drift is None:
                raise ValueError(
                    "Must specify location(s) and strength(s) of point drift terms."
                )
            self.point_log_drift = True
            point_log = np.atleast_2d(np.squeeze(np.array(point_drift, copy=True)))
            self.point_log_array = np.zeros(point_log.shape)
            self.point_log_array[:, 2] = point_log[:, 2]
            self.point_log_array[:, :2] = _adjust_for_anisotropy(
                np.vstack((point_log[:, 0], point_log[:, 1])).T,
                [self.XCENTER, self.YCENTER],
                [self.anisotropy_scaling],
                [self.anisotropy_angle],
            )
            if self.verbose:
                print(
                    "Implementing external point-logarithmic drift; "
                    "number of points =",
                    self.point_log_array.shape[0],
                    "\n",
                )
        else:
            self.point_log_drift = False

        if "specified" in drift_terms:
            if type(specified_drift) is not list:
                raise TypeError(
                    "Arrays for specified drift terms must be "
                    "encapsulated in a list."
                )
            if len(specified_drift) == 0:
                raise ValueError(
                    "Must provide at least one drift-value array "
                    "when using the 'specified' drift capability."
                )
            self.specified_drift = True
            self.specified_drift_data_arrays = []
            for term in specified_drift:
                specified = np.squeeze(np.array(term, copy=True))
                if specified.size != self.X_ORIG.size:
                    raise ValueError(
                        "Must specify the drift values for each "
                        "data point when using the 'specified' "
                        "drift capability."
                    )
                self.specified_drift_data_arrays.append(specified)
        else:
            self.specified_drift = False

        # The provided callable functions will be evaluated using
        # the adjusted coordinates.
        if "functional" in drift_terms:
            if type(functional_drift) is not list:
                raise TypeError(
                    "Callables for functional drift terms must "
                    "be encapsulated in a list."
                )
            if len(functional_drift) == 0:
                raise ValueError(
                    "Must provide at least one callable object "
                    "when using the 'functional' drift capability."
                )
            self.functional_drift = True
            self.functional_drift_terms = functional_drift
        else:
            self.functional_drift = False

    def _calculate_data_point_zscalars(self, x, y, type_="array"):
        """Determines the Z-scalar values at the specified coordinates
        for use when setting up the kriging matrix. Uses bilinear
        interpolation.
        Currently, the Z scalar values are extracted from the input Z grid
        exactly at the specified coordinates. This means that if the Z grid
        resolution is finer than the resolution of the desired kriged grid,
        there is no averaging of the scalar values to return an average
        Z value for that cell in the kriged grid. Rather, the exact Z value
        right at the coordinate is used."""

        if type_ == "scalar":
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

                if type_ == "scalar":
                    xn = x
                    yn = y
                else:
                    if x.ndim == 1:
                        xn = x[n]
                        yn = y[n]
                    else:
                        xn = x[m, n]
                        yn = y[m, n]

                if (
                    xn > np.amax(self.external_Z_array_x)
                    or xn < np.amin(self.external_Z_array_x)
                    or yn > np.amax(self.external_Z_array_y)
                    or yn < np.amin(self.external_Z_array_y)
                ):
                    raise ValueError(
                        "External drift array does not cover "
                        "specified kriging domain."
                    )

                # bilinear interpolation
                external_x2_index = np.amin(np.where(self.external_Z_array_x >= xn)[0])
                external_x1_index = np.amax(np.where(self.external_Z_array_x <= xn)[0])
                external_y2_index = np.amin(np.where(self.external_Z_array_y >= yn)[0])
                external_y1_index = np.amax(np.where(self.external_Z_array_y <= yn)[0])
                if external_y1_index == external_y2_index:
                    if external_x1_index == external_x2_index:
                        z = self.external_Z_array[external_y1_index, external_x1_index]
                    else:
                        z = (
                            self.external_Z_array[external_y1_index, external_x1_index]
                            * (self.external_Z_array_x[external_x2_index] - xn)
                            + self.external_Z_array[
                                external_y2_index, external_x2_index
                            ]
                            * (xn - self.external_Z_array_x[external_x1_index])
                        ) / (
                            self.external_Z_array_x[external_x2_index]
                            - self.external_Z_array_x[external_x1_index]
                        )
                elif external_x1_index == external_x2_index:
                    if external_y1_index == external_y2_index:
                        z = self.external_Z_array[external_y1_index, external_x1_index]
                    else:
                        z = (
                            self.external_Z_array[external_y1_index, external_x1_index]
                            * (self.external_Z_array_y[external_y2_index] - yn)
                            + self.external_Z_array[
                                external_y2_index, external_x2_index
                            ]
                            * (yn - self.external_Z_array_y[external_y1_index])
                        ) / (
                            self.external_Z_array_y[external_y2_index]
                            - self.external_Z_array_y[external_y1_index]
                        )
                else:
                    z = (
                        self.external_Z_array[external_y1_index, external_x1_index]
                        * (self.external_Z_array_x[external_x2_index] - xn)
                        * (self.external_Z_array_y[external_y2_index] - yn)
                        + self.external_Z_array[external_y1_index, external_x2_index]
                        * (xn - self.external_Z_array_x[external_x1_index])
                        * (self.external_Z_array_y[external_y2_index] - yn)
                        + self.external_Z_array[external_y2_index, external_x1_index]
                        * (self.external_Z_array_x[external_x2_index] - xn)
                        * (yn - self.external_Z_array_y[external_y1_index])
                        + self.external_Z_array[external_y2_index, external_x2_index]
                        * (xn - self.external_Z_array_x[external_x1_index])
                        * (yn - self.external_Z_array_y[external_y1_index])
                    ) / (
                        (
                            self.external_Z_array_x[external_x2_index]
                            - self.external_Z_array_x[external_x1_index]
                        )
                        * (
                            self.external_Z_array_y[external_y2_index]
                            - self.external_Z_array_y[external_y1_index]
                        )
                    )

                if type_ == "scalar":
                    z_scalars = z
                else:
                    if z_scalars.ndim == 1:
                        z_scalars[n] = z
                    else:
                        z_scalars[m, n] = z

        return z_scalars

    def update_variogram_model(
        self,
        variogram_model,
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
    ):
        """Allows user to update variogram type and/or
        variogram model parameters.

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
            Number of averaging bins for the semivariogram. Defualt is 6.
        weight : boolean, optional
            Flag that specifies if semivariance at smaller lags should be
            weighted more heavily when automatically calculating the
            variogram model. See above for more information. True indicates
            that weights will be applied. Default is False.
        anisotropy_scaling : float, optional
            Scalar stretching value to take into account anisotropy.
            Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction.
        anisotropy_angle : float, optional
            CCW angle (in degrees) by which to rotate coordinate system in
            order to take into account anisotropy. Default is 0 (no rotation).
        """

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None
        # check if a GSTools covariance model is given
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            if self.model.dim == 3:
                raise ValueError("GSTools: model dim is not 1 or 2")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling = self.model.pykrige_anis
            anisotropy_angle = self.model.pykrige_angle
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
            anisotropy_scaling != self.anisotropy_scaling
            or anisotropy_angle != self.anisotropy_angle
        ):
            if self.verbose:
                print("Adjusting data for anisotropy...")
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            self.X_ADJUSTED, self.Y_ADJUSTED = _adjust_for_anisotropy(
                np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                [self.XCENTER, self.YCENTER],
                [self.anisotropy_scaling],
                [self.anisotropy_angle],
            ).T

        if self.verbose:
            print("Updating variogram mode...")

        # See note above about the 'use_psill' kwarg...
        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
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
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
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

    def get_variogram_points(self):
        """Returns both the lags and the variogram function evaluated at each
        of them.

        The evaluation of the variogram function and the lags are produced
        internally. This method is convenient when the user wants to access to
        the lags and the resulting variogram (according to the model provided)
        for further analysis.

        Returns
        -------
        (tuple) tuple containing:
            lags (array) - the lags at which the variogram was evaluated
            variogram (array) - the variogram function evaluated at the lags
        """
        return (
            self.lags,
            self.variogram_function(self.variogram_model_parameters, self.lags),
        )

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
        """Returns the Q1, Q2, and cR statistics for the variogram fit
        (in that order). No arguments.
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

    def _get_kriging_matrix(self, n, n_withdrifts):
        """Assembles the kriging matrix."""

        xy = np.concatenate(
            (self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1
        )
        d = cdist(xy, xy, "euclidean")
        if self.UNBIAS:
            a = np.zeros((n_withdrifts + 1, n_withdrifts + 1))
        else:
            a = np.zeros((n_withdrifts, n_withdrifts))
        a[:n, :n] = -self.variogram_function(self.variogram_model_parameters, d)

        np.fill_diagonal(a, 0.0)

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
                log_dist = np.log(
                    np.sqrt(
                        (self.X_ADJUSTED - self.point_log_array[well_no, 0]) ** 2
                        + (self.Y_ADJUSTED - self.point_log_array[well_no, 1]) ** 2
                    )
                )
                if np.any(np.isinf(log_dist)):
                    log_dist[np.isinf(log_dist)] = -100.0
                a[:n, i] = -self.point_log_array[well_no, 2] * log_dist
                a[i, :n] = -self.point_log_array[well_no, 2] * log_dist
                i += 1
        if self.external_Z_drift:
            a[:n, i] = self.z_scalars
            a[i, :n] = self.z_scalars
            i += 1
        if self.specified_drift:
            for arr in self.specified_drift_data_arrays:
                a[:n, i] = arr
                a[i, :n] = arr
                i += 1
        if self.functional_drift:
            for func in self.functional_drift_terms:
                a[:n, i] = func(self.X_ADJUSTED, self.Y_ADJUSTED)
                a[i, :n] = func(self.X_ADJUSTED, self.Y_ADJUSTED)
                i += 1
        if i != n_withdrifts:
            warnings.warn(
                "Error in creating kriging matrix. Kriging may fail.", RuntimeWarning
            )
        if self.UNBIAS:
            a[n_withdrifts, :n] = 1.0
            a[:n, n_withdrifts] = 1.0
            a[n : n_withdrifts + 1, n : n_withdrifts + 1] = 0.0

        return a

    def _exec_vector(self, a, bd, xy, xy_orig, mask, n_withdrifts, spec_drift_grids):
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

        if self.UNBIAS:
            b = np.zeros((npt, n_withdrifts + 1, 1))
        else:
            b = np.zeros((npt, n_withdrifts, 1))
        b[:, :n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value and self.exact_values:
            b[zero_index[0], zero_index[1], 0] = 0.0

        i = n
        if self.regional_linear_drift:
            b[:, i, 0] = xy[:, 0]
            i += 1
            b[:, i, 0] = xy[:, 1]
            i += 1
        if self.point_log_drift:
            for well_no in range(self.point_log_array.shape[0]):
                log_dist = np.log(
                    np.sqrt(
                        (xy[:, 0] - self.point_log_array[well_no, 0]) ** 2
                        + (xy[:, 1] - self.point_log_array[well_no, 1]) ** 2
                    )
                )
                if np.any(np.isinf(log_dist)):
                    log_dist[np.isinf(log_dist)] = -100.0
                b[:, i, 0] = -self.point_log_array[well_no, 2] * log_dist
                i += 1
        if self.external_Z_drift:
            b[:, i, 0] = self._calculate_data_point_zscalars(
                xy_orig[:, 0], xy_orig[:, 1]
            )
            i += 1
        if self.specified_drift:
            for spec_vals in spec_drift_grids:
                b[:, i, 0] = spec_vals.flatten()
                i += 1
        if self.functional_drift:
            for func in self.functional_drift_terms:
                b[:, i, 0] = func(xy[:, 0], xy[:, 1])
                i += 1
        if i != n_withdrifts:
            warnings.warn(
                "Error in setting up kriging system. Kriging may fail.", RuntimeWarning,
            )
        if self.UNBIAS:
            b[:, n_withdrifts, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(
                mask[:, np.newaxis, np.newaxis], n_withdrifts + 1, axis=1
            )
            b = np.ma.array(b, mask=mask_b)

        if self.UNBIAS:
            x = (
                np.dot(a_inv, b.reshape((npt, n_withdrifts + 1)).T)
                .reshape((1, n_withdrifts + 1, npt))
                .T
            )
        else:
            x = (
                np.dot(a_inv, b.reshape((npt, n_withdrifts)).T)
                .reshape((1, n_withdrifts, npt))
                .T
            )
        zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    def _exec_loop(self, a, bd_all, xy, xy_orig, mask, n_withdrifts, spec_drift_grids):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zvalues = np.zeros(npt)
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
                zero_index = None
                zero_value = False

            if self.UNBIAS:
                b = np.zeros((n_withdrifts + 1, 1))
            else:
                b = np.zeros((n_withdrifts, 1))
            b[:n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value and self.exact_values:
                b[zero_index[0], 0] = 0.0

            i = n
            if self.regional_linear_drift:
                b[i, 0] = xy[j, 0]
                i += 1
                b[i, 0] = xy[j, 1]
                i += 1
            if self.point_log_drift:
                for well_no in range(self.point_log_array.shape[0]):
                    log_dist = np.log(
                        np.sqrt(
                            (xy[j, 0] - self.point_log_array[well_no, 0]) ** 2
                            + (xy[j, 1] - self.point_log_array[well_no, 1]) ** 2
                        )
                    )
                    if np.any(np.isinf(log_dist)):
                        log_dist[np.isinf(log_dist)] = -100.0
                    b[i, 0] = -self.point_log_array[well_no, 2] * log_dist
                    i += 1
            if self.external_Z_drift:
                b[i, 0] = self._calculate_data_point_zscalars(
                    xy_orig[j, 0], xy_orig[j, 1], type_="scalar"
                )
                i += 1
            if self.specified_drift:
                for spec_vals in spec_drift_grids:
                    b[i, 0] = spec_vals.flatten()[i]
                    i += 1
            if self.functional_drift:
                for func in self.functional_drift_terms:
                    b[i, 0] = func(xy[j, 0], xy[j, 1])
                    i += 1
            if i != n_withdrifts:
                warnings.warn(
                    "Error in setting up kriging system. Kriging may fail.",
                    RuntimeWarning,
                )
            if self.UNBIAS:
                b[n_withdrifts, 0] = 1.0

            x = np.dot(a_inv, b)
            zvalues[j] = np.sum(x[:n, 0] * self.Z)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        return zvalues, sigmasq

    def execute(
        self,
        style,
        xpoints,
        ypoints,
        mask=None,
        backend="vectorized",
        specified_drift_arrays=None,
    ):
        """Calculates a kriged grid and the associated variance.
        Includes drift terms.

        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points. Specifying 'grid'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid. Specifying 'points' treats xpoints
            and ypoints as two arrays that provide coordinate pairs at which
            to solve the kriging system. Specifying 'masked' treats xpoints and
            ypoints as two arrays of x and y coordinates that define a
            rectangular grid and uses mask to only evaluate specific points
            in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked', x-coordinates of
            MxN grid. If style is specified as 'points', x-coordinates of
            specific points at which to solve kriging system.
        ypoints : array-like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked', y-coordinates of
            MxN grid. If style is specified as 'points', y-coordinates of
            specific points at which to solve kriging system.
            Note that in this case, xpoints and ypoints must have the same
            dimensions (i.e., M = N).
        mask : boolean array, shape (M, N), optional
            Specifies the points in the rectangular grid defined by xpoints and
            ypoints that are to be excluded in the kriging calculations.
            Must be provided if style is specified as 'masked'. False indicates
            that the point should not be masked, so the kriging system will be
            solved at the point. True indicates that the point should be masked,
            so the kriging system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging. Specifying 'vectorized'
            will solve the entire kriging problem at once in a vectorized
            operation. This approach is faster but also can consume a
            significant amount of memory for large grids and/or large datasets.
            Specifying 'loop' will loop through each point at which the kriging
            system is to be solved. This approach is slower but also less
            memory-intensive. Default is 'vectorized'.
            Note that Cython backend is not supported for UK.
        specified_drift_arrays : list of array-like objects, optional
            Specifies the drift values at the points at which the kriging
            system is to be evaluated. Required if 'specified' drift provided
            in the list of drift terms when instantiating the UniversalKriging
            class. Must be a list of arrays in the same order as the list
            provided when instantiating the kriging object. Array(s) must be
            the same dimension as the specified grid or have the same number of
            points as the specified points; i.e., the arrays either must be
            shape (M, N), where M is the number of y grid-points and N is the
            number of x grid-points, or shape (M, ) or (N, 1), where M is the
            number of  points at which to evaluate the kriging system.

        Returns
        -------
        zvalues : ndarray, shape (M, N) or (N, 1)
            Z-values of specified grid or at the specified set of points.
            If style was specified as 'masked', zvalues will be a numpy
            masked array.
        sigmasq : ndarray, shape (M, N) or (N, 1)
            Variance at specified grid points or at the specified set of points.
            If style was specified as 'masked', sigmasq will be a numpy
            masked array.
        """

        if self.verbose:
            print("Executing Universal Kriging...\n")

        if style != "grid" and style != "masked" and style != "points":
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        n = self.X_ADJUSTED.shape[0]
        n_withdrifts = n
        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        nx = xpts.size
        ny = ypts.size
        if self.regional_linear_drift:
            n_withdrifts += 2
        if self.point_log_drift:
            n_withdrifts += self.point_log_array.shape[0]
        if self.external_Z_drift:
            n_withdrifts += 1
        if self.specified_drift:
            n_withdrifts += len(self.specified_drift_data_arrays)
        if self.functional_drift:
            n_withdrifts += len(self.functional_drift_terms)
        a = self._get_kriging_matrix(n, n_withdrifts)

        if style in ["grid", "masked"]:
            if style == "masked":
                if mask is None:
                    raise IOError(
                        "Must specify boolean masking array when style is 'masked'."
                    )
                if mask.shape[0] != ny or mask.shape[1] != nx:
                    if mask.shape[0] == nx and mask.shape[1] == ny:
                        mask = mask.T
                    else:
                        raise ValueError(
                            "Mask dimensions do not match specified grid dimensions."
                        )
                mask = mask.flatten()
            npt = ny * nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()

        elif style == "points":
            if xpts.size != ypts.size:
                raise ValueError(
                    "xpoints and ypoints must have same "
                    "dimensions when treated as listing "
                    "discrete points."
                )
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        if specified_drift_arrays is None:
            specified_drift_arrays = []
        spec_drift_grids = []
        if self.specified_drift:
            if len(specified_drift_arrays) == 0:
                raise ValueError(
                    "Must provide drift values for kriging points "
                    "when using 'specified' drift capability."
                )
            if type(specified_drift_arrays) is not list:
                raise TypeError(
                    "Arrays for specified drift terms must be "
                    "encapsulated in a list."
                )
            for spec in specified_drift_arrays:
                if style in ["grid", "masked"]:
                    if spec.ndim < 2:
                        raise ValueError(
                            "Dimensions of drift values array do "
                            "not match specified grid dimensions."
                        )
                    elif spec.shape[0] != ny or spec.shape[1] != nx:
                        if spec.shape[0] == nx and spec.shape[1] == ny:
                            spec_drift_grids.append(np.squeeze(spec.T))
                        else:
                            raise ValueError(
                                "Dimensions of drift values array "
                                "do not match specified grid "
                                "dimensions."
                            )
                    else:
                        spec_drift_grids.append(np.squeeze(spec))
                elif style == "points":
                    if spec.ndim != 1:
                        raise ValueError(
                            "Dimensions of drift values array do "
                            "not match specified grid dimensions."
                        )
                    elif spec.shape[0] != xpts.size:
                        raise ValueError(
                            "Number of supplied drift values in "
                            "array do not match specified number "
                            "of kriging points."
                        )
                    else:
                        spec_drift_grids.append(np.squeeze(spec))
            if len(spec_drift_grids) != len(self.specified_drift_data_arrays):
                raise ValueError(
                    "Inconsistent number of specified drift terms supplied."
                )
        else:
            if len(specified_drift_arrays) != 0:
                warnings.warn(
                    "Provided specified drift values, but "
                    "'specified' drift was not initialized during "
                    "instantiation of UniversalKriging class.",
                    RuntimeWarning,
                )

        xy_points_original = np.concatenate(
            (xpts[:, np.newaxis], ypts[:, np.newaxis]), axis=1
        )
        xpts, ypts = _adjust_for_anisotropy(
            np.vstack((xpts, ypts)).T,
            [self.XCENTER, self.YCENTER],
            [self.anisotropy_scaling],
            [self.anisotropy_angle],
        ).T
        xy_points = np.concatenate((xpts[:, np.newaxis], ypts[:, np.newaxis]), axis=1)
        xy_data = np.concatenate(
            (self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1
        )

        if style != "masked":
            mask = np.zeros(npt, dtype="bool")

        bd = cdist(xy_points, xy_data, "euclidean")
        if backend == "vectorized":
            zvalues, sigmasq = self._exec_vector(
                a,
                bd,
                xy_points,
                xy_points_original,
                mask,
                n_withdrifts,
                spec_drift_grids,
            )
        elif backend == "loop":
            zvalues, sigmasq = self._exec_loop(
                a,
                bd,
                xy_points,
                xy_points_original,
                mask,
                n_withdrifts,
                spec_drift_grids,
            )
        else:
            raise ValueError(
                "Specified backend {} is not supported "
                "for 2D universal kriging.".format(backend)
            )

        if style == "masked":
            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ["masked", "grid"]:
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq
