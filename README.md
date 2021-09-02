# PyKrige

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3738604.svg)](https://doi.org/10.5281/zenodo.3738604)
[![PyPI version](https://badge.fury.io/py/PyKrige.svg)](https://badge.fury.io/py/PyKrige)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pykrige.svg)](https://anaconda.org/conda-forge/pykrige)
[![Build Status](https://github.com/GeoStat-Framework/PyKrige/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/GeoStat-Framework/PyKrige/actions)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/PyKrige/badge.svg?branch=main)](https://coveralls.io/github/GeoStat-Framework/PyKrige?branch=main)
[![Documentation Status](https://readthedocs.org/projects/pykrige/badge/?version=stable)](http://pykrige.readthedocs.io/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<p align="center">
<img src="https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/docs/source/pics/PyKrige_250.png" alt="PyKrige-LOGO" width="251px"/>
</p>

Kriging Toolkit for Python.

## Purpose

The code supports 2D and 3D ordinary and universal kriging. Standard
variogram models (linear, power, spherical, gaussian, exponential) are
built in, but custom variogram models can also be used. The 2D universal
kriging code currently supports regional-linear, point-logarithmic, and
external drift terms, while the 3D universal kriging code supports a
regional-linear drift term in all three spatial dimensions. Both
universal kriging classes also support generic 'specified' and
'functional' drift capabilities. With the 'specified' drift capability,
the user may manually specify the values of the drift(s) at each data
point and all grid points. With the 'functional' drift capability, the
user may provide callable function(s) of the spatial coordinates that
define the drift(s). The package includes a module that contains
functions that should be useful in working with ASCII grid files (`\*.asc`).

See the documentation at <http://pykrige.readthedocs.io/> for more
details and examples.

## Installation

PyKrige requires Python 3.5+ as well as numpy, scipy. It can be
installed from PyPi with,

``` bash
pip install pykrige
```

scikit-learn is an optional dependency needed for parameter tuning and
regression kriging. matplotlib is an optional dependency needed for
plotting.

If you use conda, PyKrige can be installed from the <span
class="title-ref">conda-forge</span> channel with,

``` bash
conda install -c conda-forge pykrige
```

## Features

### Kriging algorithms

-   `OrdinaryKriging`: 2D ordinary kriging with estimated mean
-   `UniversalKriging`: 2D universal kriging providing drift terms
-   `OrdinaryKriging3D`: 3D ordinary kriging
-   `UniversalKriging3D`: 3D universal kriging
-   `RegressionKriging`: An implementation of Regression-Kriging
-   `ClassificationKriging`: An implementation of Simplicial Indicator
    Kriging

### Wrappers

-   `rk.Krige`: A scikit-learn wrapper class for Ordinary and Universal
    Kriging

### Tools

-   `kriging_tools.write_asc_grid`: Writes gridded data to ASCII grid file (`\*.asc`)
-   `kriging_tools.read_asc_grid`: Reads ASCII grid file (`\*.asc`)
-   `kriging_tools.write_zmap_grid`: Writes gridded data to zmap file (`\*.zmap`)
-   `kriging_tools.read_zmap_grid`: Reads zmap file (`\*.zmap`)

### Kriging Parameters Tuning

A scikit-learn compatible API for parameter tuning by cross-validation
is exposed in
[sklearn.model\_selection.GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
See the [Krige
CV](http://pykrige.readthedocs.io/en/latest/examples/08_krige_cv.html#sphx-glr-examples-08-krige-cv-py)
example for a more practical illustration.

### Regression Kriging

[Regression kriging](https://en.wikipedia.org/wiki/Regression-Kriging)
can be performed with
[pykrige.rk.RegressionKriging](http://pykrige.readthedocs.io/en/latest/examples/07_regression_kriging2d.html).
This class takes as parameters a scikit-learn regression model, and
details of either the `OrdinaryKriging` or the `UniversalKriging`
class, and performs a correction step on the ML regression prediction.

A demonstration of the regression kriging is provided in the
[corresponding
example](http://pykrige.readthedocs.io/en/latest/examples/07_regression_kriging2d.html#sphx-glr-examples-07-regression-kriging2d-py).

### Classification Kriging

[Simplifical Indicator
kriging](https://www.sciencedirect.com/science/article/abs/pii/S1002070508600254)
can be performed with
[pykrige.ck.ClassificationKriging](http://pykrige.readthedocs.io/en/latest/examples/10_classification_kriging2d.html).
This class takes as parameters a scikit-learn classification model, and
details of either the `OrdinaryKriging` or the `UniversalKriging` class,
and performs a correction step on the ML classification prediction.

A demonstration of the classification kriging is provided in the
[corresponding
example](http://pykrige.readthedocs.io/en/latest/examples/10_classification_kriging2d.html#sphx-glr-examples-10-classification-kriging2d-py).

## License

PyKrige uses the BSD 3-Clause License.
