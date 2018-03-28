from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

__doc__ = """
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

Copyright (c) 2015-2018, PyKrige Developers
"""


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
    return scale * d**exponent + nugget


def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget


def exponential_variogram_model(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d/(range_/3.))) + nugget


def spherical_variogram_model(m, d):
    """Spherical model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget])


def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - (1.-d/(range_/3.)) * np.exp(-d/(range_/3.))) + nugget
