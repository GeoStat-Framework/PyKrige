# coding: utf-8
"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Function definitions for variogram models. In each function, m is a list of
defining parameters and d is an array of the distance values at which to
calculate the variogram model.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2020, PyKrige Developers
"""
import numpy as np


def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def power_variogram_model(m, d):
    """Power model, m is [scale, exponent, nugget]"""
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d ** exponent + nugget


def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-(d ** 2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget


def exponential_variogram_model(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget


def spherical_variogram_model(m, d):
    """Spherical model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(
        d,
        [d <= range_, d > range_],
        [
            lambda x: psill
            * ((3.0 * x) / (2.0 * range_) - (x ** 3.0) / (2.0 * range_ ** 3.0))
            + nugget,
            psill + nugget,
        ],
    )


def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return (
        psill * (1.0 - (1.0 - d / (range_ / 3.0)) * np.exp(-d / (range_ / 3.0)))
        + nugget
    )
