PyKrige
=======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3738604.svg
   :target: https://doi.org/10.5281/zenodo.3738604
.. image:: https://badge.fury.io/py/PyKrige.svg
   :target: https://badge.fury.io/py/PyKrige
.. image:: https://img.shields.io/conda/vn/conda-forge/pykrige.svg
   :target: https://anaconda.org/conda-forge/pykrige
.. image:: https://travis-ci.com/GeoStat-Framework/PyKrige.svg?branch=master
   :target: https://travis-ci.com/GeoStat-Framework/PyKrige
.. image:: https://coveralls.io/repos/github/GeoStat-Framework/PyKrige/badge.svg?branch=master
   :target: https://coveralls.io/github/GeoStat-Framework/PyKrige?branch=master
.. image:: https://readthedocs.org/projects/pykrige/badge/?version=stable
   :target: http://pykrige.readthedocs.io/en/stable/?badge=stable
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black


.. figure:: https://github.com/GeoStat-Framework/GeoStat-Framework.github.io/raw/master/docs/source/pics/PyKrige_250.png
   :align: center
   :alt: PyKrige
   :figclass: align-center


Kriging Toolkit for Python.


Purpose
^^^^^^^

The code supports 2D and 3D ordinary and universal kriging. Standard variogram models
(linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used.
The 2D universal kriging code currently supports regional-linear, point-logarithmic, and external drift terms,
while the 3D universal kriging code supports a regional-linear drift term in all three spatial dimensions.
Both universal kriging classes also support generic 'specified' and 'functional' drift capabilities.
With the 'specified' drift capability, the user may manually specify the values of the drift(s) at each data
point and all grid points. With the 'functional' drift capability, the user may provide callable function(s)
of the spatial coordinates that define the drift(s). The package includes a module that contains functions
that should be useful in working with ASCII grid files (`*.asc`).

See the documentation at `http://pykrige.readthedocs.io/ <http://pykrige.readthedocs.io/>`_
for more details and examples.


Installation
^^^^^^^^^^^^

PyKrige requires Python 3.5+ as well as numpy, scipy. It can be installed from PyPi with,

.. code:: bash

    pip install pykrige

scikit-learn is an optional dependency needed for parameter tuning and regression kriging.
matplotlib is an optional dependency needed for plotting.

If you use conda, PyKrige can be installed from the `conda-forge` channel with,

.. code:: bash

    conda install -c conda-forge pykrige


Features
^^^^^^^^

Kriging algorithms
------------------

* ``OrdinaryKriging``: 2D ordinary kriging with estimated mean
* ``UniversalKriging``: 2D universal kriging providing drift terms
* ``OrdinaryKriging3D``: 3D ordinary kriging
* ``UniversalKriging3D``: 3D universal kriging
* ``RegressionKriging``: An implementation of Regression-Kriging


Wrappers
--------

* ``rk.Krige``: A scikit-learn wrapper class for Ordinary and Universal Kriging


Tools
-----

* ``kriging_tools.write_asc_grid``: Writes gridded data to ASCII grid file (\*.asc)
* ``kriging_tools.read_asc_grid``: Reads ASCII grid file (\*.asc)


Kriging Parameters Tuning
-------------------------

A scikit-learn compatible API for parameter tuning by cross-validation is exposed in
`sklearn.model_selection.GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
See the `Krige CV <http://pykrige.readthedocs.io/en/latest/examples/08_krige_cv.html#sphx-glr-examples-08-krige-cv-py>`_
example for a more practical illustration.


Regression Kriging
------------------

`Regression kriging <https://en.wikipedia.org/wiki/Regression-Kriging>`_ can be performed
with `pykrige.rk.RegressionKriging <http://pykrige.readthedocs.io/en/latest/examples/07_regression_kriging2d.html>`_.
This class takes as parameters a scikit-learn regression model, and details of either either
the ``OrdinaryKriging`` or the ``UniversalKriging`` class, and performs a correction steps on the ML regression prediction.

A demonstration of the regression kriging is provided in the
`corresponding example <http://pykrige.readthedocs.io/en/latest/examples/07_regression_kriging2d.html#sphx-glr-examples-07-regression-kriging2d-py>`_.


License
^^^^^^^

PyKrige uses the BSD 3-Clause License.
