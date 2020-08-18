# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Methods used by multiple classes.

References
----------
[1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

[2] T. Vincenty, Direct and Inverse Solutions of Geodesics on the Ellipsoid
    with Application of Nested Equations, Survey Review 23 (176),
    (Directorate of Overseas Survey, Kingston Road, Tolworth, Surrey 1975)

Copyright (c) 2015-2020, PyKrige Developers
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import least_squares
import scipy.linalg as spl


eps = 1.0e-10  # Cutoff for comparison to zero


P_INV = {"pinv": spl.pinv, "pinv2": spl.pinv2, "pinvh": spl.pinvh}


def great_circle_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between one or multiple pairs of
    points given in spherical coordinates. Spherical coordinates are expected
    in degrees. Angle definition follows standard longitude/latitude definition.
    This uses the arctan version of the great-circle distance function
    (en.wikipedia.org/wiki/Great-circle_distance) for increased
    numerical stability.

    Parameters
    ----------
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


    Returns
    -------
    distance: float scalar or numpy array
        The great circle distance(s) (in degrees) between the
        given pair(s) of points.

    """
    # Convert to radians:
    lat1 = np.array(lat1) * np.pi / 180.0
    lat2 = np.array(lat2) * np.pi / 180.0
    dlon = (lon1 - lon2) * np.pi / 180.0

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

    return (
        180.0
        / np.pi
        * np.arctan2(
            np.sqrt((c2 * np.sin(dlon)) ** 2 + (c1 * s2 - s1 * c2 * cd) ** 2),
            s1 * s2 + c1 * c2 * cd,
        )
    )


def euclid3_to_great_circle(euclid3_distance):
    """Convert euclidean distance between points on a unit sphere to
    the corresponding great circle distance.

    Parameters
    ----------
    euclid3_distance: float scalar or numpy array
        The euclidean three-space distance(s) between points on a
        unit sphere, thus between [0,2].

    Returns
    -------
    great_circle_dist: float scalar or numpy array
        The corresponding great circle distance(s) between the points.
    """
    # Eliminate some possible numerical errors:
    euclid3_distance[euclid3_distance > 2.0] = 2.0
    return 180.0 - 360.0 / np.pi * np.arccos(0.5 * euclid3_distance)


def _adjust_for_anisotropy(X, center, scaling, angle):
    """Adjusts data coordinates to take into account anisotropy.
    Can also be used to take into account data scaling. Angles are CCW about
    specified axes. Scaling is applied in rotated coordinate system.

    Parameters
    ----------
    X : ndarray
        float array [n_samples, n_dim], the input array of coordinates
    center : ndarray
        float array [n_dim], the coordinate of centers
    scaling : ndarray
        float array [n_dim - 1], the scaling of last two dimensions
    angle : ndarray
        float array [2*n_dim - 3], the anisotropy angle (degrees)

    Returns
    -------
    X_adj : ndarray
        float array [n_samples, n_dim], the X array adjusted for anisotropy.
    """

    center = np.asarray(center)[None, :]
    angle = np.asarray(angle) * np.pi / 180

    X -= center

    Ndim = X.shape[1]

    if Ndim == 1:
        raise NotImplementedError("Not implemnented yet?")
    elif Ndim == 2:
        stretch = np.array([[1, 0], [0, scaling[0]]])
        rot_tot = np.array(
            [
                [np.cos(-angle[0]), -np.sin(-angle[0])],
                [np.sin(-angle[0]), np.cos(-angle[0])],
            ]
        )
    elif Ndim == 3:
        stretch = np.array(
            [[1.0, 0.0, 0.0], [0.0, scaling[0], 0.0], [0.0, 0.0, scaling[1]]]
        )
        rotate_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(-angle[0]), -np.sin(-angle[0])],
                [0.0, np.sin(-angle[0]), np.cos(-angle[0])],
            ]
        )
        rotate_y = np.array(
            [
                [np.cos(-angle[1]), 0.0, np.sin(-angle[1])],
                [0.0, 1.0, 0.0],
                [-np.sin(-angle[1]), 0.0, np.cos(-angle[1])],
            ]
        )
        rotate_z = np.array(
            [
                [np.cos(-angle[2]), -np.sin(-angle[2]), 0.0],
                [np.sin(-angle[2]), np.cos(-angle[2]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot_tot = np.dot(rotate_z, np.dot(rotate_y, rotate_x))
    else:
        raise ValueError(
            "Adjust for anisotropy function doesn't support ND spaces where N>3"
        )
    X_adj = np.dot(stretch, np.dot(rot_tot, X.T)).T

    X_adj += center

    return X_adj


def _make_variogram_parameter_list(variogram_model, variogram_model_parameters):
    """Converts the user input for the variogram model parameters into the
    format expected in the rest of the code.

    Makes a list of variogram model parameters in the expected order if the
    user has provided the model parameters. If not, returns None, which
    will ensure that the automatic variogram estimation routine is
    triggered.

    Parameters
    ----------
    variogram_model : str
        specifies the variogram model type
    variogram_model_parameters : list, dict, or None
        parameters provided by the user, can also be None if the user
        did not specify the variogram model parameters; if None,
        this function returns None, that way the automatic variogram
        estimation routine will kick in down the road...

    Returns
    -------
    parameter_list : list
        variogram model parameters stored in a list in the expected order;
        if variogram_model is 'custom', model parameters should already
        be encapsulated in a list, so the list is returned unaltered;
        if variogram_model_parameters was not specified by the user,
        None is returned; order for internal variogram models is as follows...

        linear - [slope, nugget]
        power - [scale, exponent, nugget]
        gaussian - [psill, range, nugget]
        spherical - [psill, range, nugget]
        exponential - [psill, range, nugget]
        hole-effect - [psill, range, nugget]

    """

    if variogram_model_parameters is None:

        parameter_list = None

    elif type(variogram_model_parameters) is dict:

        if variogram_model in ["linear"]:

            if (
                "slope" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'linear' variogram model requires 'slope' "
                    "and 'nugget' specified in variogram model "
                    "parameter dictionary."
                )

            else:

                parameter_list = [
                    variogram_model_parameters["slope"],
                    variogram_model_parameters["nugget"],
                ]

        elif variogram_model in ["power"]:

            if (
                "scale" not in variogram_model_parameters.keys()
                or "exponent" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'power' variogram model requires 'scale', "
                    "'exponent', and 'nugget' specified in "
                    "variogram model parameter dictionary."
                )

            else:

                parameter_list = [
                    variogram_model_parameters["scale"],
                    variogram_model_parameters["exponent"],
                    variogram_model_parameters["nugget"],
                ]

        elif variogram_model in ["gaussian", "spherical", "exponential", "hole-effect"]:

            if (
                "range" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'%s' variogram model requires 'range', "
                    "'nugget', and either 'sill' or 'psill' "
                    "specified in variogram model parameter "
                    "dictionary." % variogram_model
                )

            else:

                if "sill" in variogram_model_parameters.keys():

                    parameter_list = [
                        variogram_model_parameters["sill"]
                        - variogram_model_parameters["nugget"],
                        variogram_model_parameters["range"],
                        variogram_model_parameters["nugget"],
                    ]

                elif "psill" in variogram_model_parameters.keys():

                    parameter_list = [
                        variogram_model_parameters["psill"],
                        variogram_model_parameters["range"],
                        variogram_model_parameters["nugget"],
                    ]

                else:

                    raise KeyError(
                        "'%s' variogram model requires either "
                        "'sill' or 'psill' specified in "
                        "variogram model parameter "
                        "dictionary." % variogram_model
                    )

        elif variogram_model in ["custom"]:

            raise TypeError(
                "For user-specified custom variogram model, "
                "parameters must be specified in a list, "
                "not a dict."
            )

        else:

            raise ValueError(
                "Specified variogram model must be one of the "
                "following: 'linear', 'power', 'gaussian', "
                "'spherical', 'exponential', 'hole-effect', "
                "'custom'."
            )

    elif type(variogram_model_parameters) is list:

        if variogram_model in ["linear"]:

            if len(variogram_model_parameters) != 2:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly two entries when variogram model "
                    "set to 'linear'."
                )

            parameter_list = variogram_model_parameters

        elif variogram_model in ["power"]:

            if len(variogram_model_parameters) != 3:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly three entries when variogram model "
                    "set to 'power'."
                )

            parameter_list = variogram_model_parameters

        elif variogram_model in ["gaussian", "spherical", "exponential", "hole-effect"]:

            if len(variogram_model_parameters) != 3:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly three entries when variogram model "
                    "set to '%s'." % variogram_model
                )

            parameter_list = [
                variogram_model_parameters[0] - variogram_model_parameters[2],
                variogram_model_parameters[1],
                variogram_model_parameters[2],
            ]

        elif variogram_model in ["custom"]:

            parameter_list = variogram_model_parameters

        else:

            raise ValueError(
                "Specified variogram model must be one of the "
                "following: 'linear', 'power', 'gaussian', "
                "'spherical', 'exponential', 'hole-effect', "
                "'custom'."
            )

    else:

        raise TypeError(
            "Variogram model parameters must be provided in either "
            "a list or a dict when they are explicitly specified."
        )

    return parameter_list


def _initialize_variogram_model(
    X,
    y,
    variogram_model,
    variogram_model_parameters,
    variogram_function,
    nlags,
    weight,
    coordinates_type,
):
    """Initializes the variogram model for kriging. If user does not specify
    parameters, calls automatic variogram estimation routine.
    Returns lags, semivariance, and variogram model parameters.

    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of values to be kriged
    variogram_model: str
        user-specified variogram model to use
    variogram_model_parameters: list
        user-specified parameters for variogram model
    variogram_function: callable
        function that will be called to evaluate variogram model
        (only used if user does not specify variogram model parameters)
    nlags: int
        integer scalar, number of bins into which to group inter-point distances
    weight: bool
        boolean flag that indicates whether the semivariances at smaller lags
        should be weighted more heavily in the automatic variogram estimation
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon

    Returns
    -------
    lags: ndarray
        float array [nlags], distance values for bins into which the
        semivariances were grouped
    semivariance: ndarray
        float array [nlags], averaged semivariance for each bin
    variogram_model_parameters: list
        parameters for the variogram model, either returned unaffected if the
        user specified them or returned from the automatic variogram
        estimation routine
    """

    # distance calculation for rectangular coords now leverages
    # scipy.spatial.distance's pdist function, which gives pairwise distances
    # in a condensed distance vector (distance matrix flattened to a vector)
    # to calculate semivariances...
    if coordinates_type == "euclidean":
        d = pdist(X, metric="euclidean")
        g = 0.5 * pdist(y[:, None], metric="sqeuclidean")

    # geographic coordinates only accepted if the problem is 2D
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # old method of distance calculation is retained here...
    # could be improved in the future
    elif coordinates_type == "geographic":
        if X.shape[1] != 2:
            raise ValueError(
                "Geographic coordinate type only supported for 2D datasets."
            )
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        z1, z2 = np.meshgrid(y, y, sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        g = 0.5 * (z1 - z2) ** 2.0
        indices = np.indices(d.shape)
        d = d[(indices[0, :, :] > indices[1, :, :])]
        g = g[(indices[0, :, :] > indices[1, :, :])]

    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    # Equal-sized bins are now implemented. The upper limit on the bins
    # is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities
    # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
    # Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin) / nlags
    bins = [dmin + n * dd for n in range(nlags)]
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

    # a few tests the make sure that, if the variogram_model_parameters
    # are supplied, they have been supplied as expected...
    # if variogram_model_parameters was not defined, then estimate the variogram
    if variogram_model_parameters is not None:
        if variogram_model == "linear" and len(variogram_model_parameters) != 2:
            raise ValueError(
                "Exactly two parameters required for linear variogram model."
            )
        elif (
            variogram_model
            in ["power", "spherical", "exponential", "gaussian", "hole-effect"]
            and len(variogram_model_parameters) != 3
        ):
            raise ValueError(
                "Exactly three parameters required for "
                "%s variogram model" % variogram_model
            )
    else:
        if variogram_model == "custom":
            raise ValueError(
                "Variogram parameters must be specified when "
                "implementing custom variogram model."
            )
        else:
            variogram_model_parameters = _calculate_variogram_model(
                lags, semivariance, variogram_model, variogram_function, weight
            )

    return lags, semivariance, variogram_model_parameters


def _variogram_residuals(params, x, y, variogram_function, weight):
    """Function used in variogram model estimation. Returns residuals between
    calculated variogram and actual data (lags/semivariance).
    Called by _calculate_variogram_model.

    Parameters
    ----------
    params: list or 1D array
        parameters for calculating the model variogram
    x: ndarray
        lags (distances) at which to evaluate the model variogram
    y: ndarray
        experimental semivariances at the specified lags
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to
        fit smaller lags better

    Returns
    -------
    resid: 1d array
        residuals, dimension same as y
    """

    # this crude weighting routine can be used to better fit the model
    # variogram to the experimental variogram at smaller lags...
    # the weights are calculated from a logistic function, so weights at small
    # lags are ~1 and weights at the longest lags are ~0;
    # the center of the logistic weighting is hard-coded to be at 70% of the
    # distance from the shortest lag to the largest lag
    if weight:
        drange = np.amax(x) - np.amin(x)
        k = 2.1972 / (0.1 * drange)
        x0 = 0.7 * drange + np.amin(x)
        weights = 1.0 / (1.0 + np.exp(-k * (x0 - x)))
        weights /= np.sum(weights)
        resid = (variogram_function(params, x) - y) * weights
    else:
        resid = variogram_function(params, x) - y

    return resid


def _calculate_variogram_model(
    lags, semivariance, variogram_model, variogram_function, weight
):
    """Function that fits a variogram model when parameters are not specified.
    Returns variogram model parameters that minimize the RMSE between the
    specified variogram function and the actual calculated variogram points.

    Parameters
    ----------
    lags: 1d array
        binned lags/distances to use for variogram model parameter estimation
    semivariance: 1d array
        binned/averaged experimental semivariances to use for variogram model
        parameter estimation
    variogram_model: str/unicode
        specified variogram model to use for parameter estimation
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to fit
        smaller lags better this is passed on to the residual calculation
        cfunction, where weighting is actually applied...

    Returns
    -------
    res: list
        list of estimated variogram model parameters

    NOTE that the estimation routine works in terms of the partial sill
    (psill = sill - nugget) -- setting bounds such that psill > 0 ensures that
    the sill will always be greater than the nugget...
    """

    if variogram_model == "linear":
        x0 = [
            (np.amax(semivariance) - np.amin(semivariance))
            / (np.amax(lags) - np.amin(lags)),
            np.amin(semivariance),
        ]
        bnds = ([0.0, 0.0], [np.inf, np.amax(semivariance)])
    elif variogram_model == "power":
        x0 = [
            (np.amax(semivariance) - np.amin(semivariance))
            / (np.amax(lags) - np.amin(lags)),
            1.1,
            np.amin(semivariance),
        ]
        bnds = ([0.0, 0.001, 0.0], [np.inf, 1.999, np.amax(semivariance)])
    else:
        x0 = [
            np.amax(semivariance) - np.amin(semivariance),
            0.25 * np.amax(lags),
            np.amin(semivariance),
        ]
        bnds = (
            [0.0, 0.0, 0.0],
            [10.0 * np.amax(semivariance), np.amax(lags), np.amax(semivariance)],
        )

    # use 'soft' L1-norm minimization in order to buffer against
    # potential outliers (weird/skewed points)
    res = least_squares(
        _variogram_residuals,
        x0,
        bounds=bnds,
        loss="soft_l1",
        args=(lags, semivariance, variogram_function, weight),
    )

    return res.x


def _krige(
    X,
    y,
    coords,
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
):
    """Sets up and solves the ordinary kriging system for the given
    coordinate pair. This function is only used for the statistics calculations.

    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of measurement values
    coords: ndarray
        float array [1, n_dim], point at which to evaluate the kriging system
    variogram_function: callable
        function that will be called to evaluate variogram model
    variogram_model_parameters: list
        user-specified parameters for variogram model
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False

    Returns
    -------
    zinterp: float
        kriging estimate at the specified point
    sigmasq: float
        mean square error of the kriging estimate
    """

    zero_index = None
    zero_value = False

    # calculate distance between points... need a square distance matrix
    # of inter-measurement-point distances and a vector of distances between
    # measurement points (X) and the kriging point (coords)
    if coordinates_type == "euclidean":
        d = squareform(pdist(X, metric="euclidean"))
        bd = np.squeeze(cdist(X, coords[None, :], metric="euclidean"))

    # geographic coordinate distances still calculated in the old way...
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # also assume problem is 2D; check done earlier in initializing variogram
    elif coordinates_type == "geographic":
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        bd = great_circle_distance(
            X[:, 0],
            X[:, 1],
            coords[0] * np.ones(X.shape[0]),
            coords[1] * np.ones(X.shape[0]),
        )

    # this check is done when initializing variogram, but kept here anyways...
    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    # check if kriging point overlaps with measurement point
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    # set up kriging matrix
    n = X.shape[0]
    a = np.zeros((n + 1, n + 1))
    a[:n, :n] = -variogram_function(variogram_model_parameters, d)
    np.fill_diagonal(a, 0.0)
    a[n, :] = 1.0
    a[:, n] = 1.0
    a[n, n] = 0.0

    # set up RHS
    b = np.zeros((n + 1, 1))
    b[:n, 0] = -variogram_function(variogram_model_parameters, bd)
    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    # solve
    if pseudo_inv:
        res = np.linalg.lstsq(a, b, rcond=None)[0]
    else:
        res = np.linalg.solve(a, b)
    zinterp = np.sum(res[:n, 0] * y)
    sigmasq = np.sum(res[:, 0] * -b[:, 0])

    return zinterp, sigmasq


def _find_statistics(
    X,
    y,
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
):
    """Calculates variogram fit statistics.
    Returns the delta, sigma, and epsilon values for the variogram fit.
    These arrays are used for statistics calculations.

    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of measurement values
    variogram_function: callable
        function that will be called to evaluate variogram model
    variogram_model_parameters: list
        user-specified parameters for variogram model
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False

    Returns
    -------
    delta: ndarray
        residuals between observed values and kriged estimates for those values
    sigma: ndarray
        mean error in kriging estimates
    epsilon: ndarray
        residuals normalized by their mean error
    """

    delta = np.zeros(y.shape)
    sigma = np.zeros(y.shape)

    for i in range(y.shape[0]):

        # skip the first value in the kriging problem
        if i == 0:
            continue

        else:
            k, ss = _krige(
                X[:i, :],
                y[:i],
                X[i, :],
                variogram_function,
                variogram_model_parameters,
                coordinates_type,
                pseudo_inv,
            )

            # if the estimation error is zero, it's probably because
            # the evaluation point X[i, :] is really close to one of the
            # kriging system points in X[:i, :]...
            # in the case of zero estimation error, the results are not stored
            if np.absolute(ss) < eps:
                continue

            delta[i] = y[i] - k
            sigma[i] = np.sqrt(ss)

    # only use non-zero entries in these arrays... sigma is used to pull out
    # non-zero entries in both cases because it is guaranteed to be positive,
    # whereas delta can be either positive or negative
    delta = delta[sigma > eps]
    sigma = sigma[sigma > eps]
    epsilon = delta / sigma

    return delta, sigma, epsilon


def calcQ1(epsilon):
    """Returns the Q1 statistic for the variogram fit (see [1])."""
    return abs(np.sum(epsilon) / (epsilon.shape[0] - 1))


def calcQ2(epsilon):
    """Returns the Q2 statistic for the variogram fit (see [1])."""
    return np.sum(epsilon ** 2) / (epsilon.shape[0] - 1)


def calc_cR(Q2, sigma):
    """Returns the cR statistic for the variogram fit (see [1])."""
    return Q2 * np.exp(np.sum(np.log(sigma ** 2)) / sigma.shape[0])
