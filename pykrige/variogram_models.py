from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__doc__ = """Code by Benjamin S. Murphy
bscott.murphy@gmail.com

Dependencies:
    numpy

Methods:

    linear_variogram_model(m, d):
        m (array-like): [slope, nugget]
        d (array-like): Points at which to calculate variogram model.

    power_variogram_model(m, d):
        m (array-like): [scale, exponent, nugget]
        d (array-like): Points at which to calculate variogram model.

    gaussian_variogram_model(m, d):
        m (array-like): [psill, range, nugget]
        d (array-like): Points at which to calculate variogram model.

    exponential_variogram_model(m, d):
        m (array-like): [psill, range, nugget]
        d (array-like): Points at which to calculate variogram model.

    spherical_variogram_model(m, d):
        m (array-like): [psill, range, nugget]
        d (array-like): Points at which to calculate variogram model.

    hole_effect_variogram_model(m, d):
        m (array-like): [psill, range, nugget]
        d (array-like): Points at which to calculate variogram model.

*** NOTE these functions use the partial sill (psill = sill - nugget) rather than the full sill...
    the PyKrige user interface by default takes the full sill (although this can be changed with a flag),
    but it's safer to perform automatic variogram estimation using the partial sill

*** ALSO NOTE that Kitanidis says the hole-effect variogram model is only correct for the 1D case...
    it's implemented here for completeness and should be used cautiously...

References:
    P.K. Kitanidis, Introduction to Geostatistcs: Applications in Hydrogeology,
    (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2017 Benjamin S. Murphy
"""

import numpy as np


def linear_variogram_model(m, d):
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def power_variogram_model(m, d):
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d**exponent + nugget


def gaussian_variogram_model(m, d):
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget


def exponential_variogram_model(m, d):
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d/(range_/3.))) + nugget


def spherical_variogram_model(m, d):
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget])


def hole_effect_variogram_model(m, d):
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - (1.-d/(range_/3.)) * np.exp(-d/(range_/3.))) + nugget
