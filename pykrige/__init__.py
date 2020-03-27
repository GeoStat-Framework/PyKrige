"""
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Kriging toolkit for Python.

ok: Contains class OrdinaryKriging, which is a convenience class for easy
    access to 2D ordinary kriging.
uk: Contains class UniversalKriging, which provides more control over
    2D kriging by utilizing drift terms. Supported drift terms currently
    include point-logarithmic, regional linear, and external z-scalar.
    Generic functions of the spatial coordinates may also be supplied to
    provide drift terms, or the point-by-point values of a drift term
    may be supplied.
ok3d: Contains class OrdinaryKriging3D, which provides support for
    3D ordinary kriging.
uk3d: Contains class UniversalKriging3D, which provide support for
    3D universal kriging. A regional linear drift is the only drift term
    currently supported, but generic drift functions or point-by-point
    values of a drift term may also be supplied.
kriging_tools: Contains a set of functions to work with *.asc files.
variogram_models: Contains the definitions for the implemented variogram
    models. Note that the utilized formulas are as presented in Kitanidis,
    so the exact definition of the range (specifically, the associated
    scaling of that value) may differ slightly from other sources.
core: Contains the backbone functions of the package that are called by both
    the various kriging classes. The functions were consolidated here
    in order to reduce redundancy in the code.
test: Contains the test script.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistics: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2020, PyKrige Developers
"""

from . import kriging_tools as kt  # noqa
from .ok import OrdinaryKriging  # noqa
from .uk import UniversalKriging  # noqa
from .ok3d import OrdinaryKriging3D  # noqa
from .uk3d import UniversalKriging3D  # noqa

try:
    from pykrige._version import __version__
except ImportError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"


__author__ = "Benjamin S. Murphy"


__all__ = ["__version__"]
__all__ += ["kt", "ok", "uk", "ok3d", "uk3d", "kriging_tools"]
__all__ += ["OrdinaryKriging"]
__all__ += ["UniversalKriging"]
__all__ += ["OrdinaryKriging3D"]
__all__ += ["UniversalKriging3D"]
