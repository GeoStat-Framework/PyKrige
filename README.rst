PyKrige
=======

Kriging Toolkit for Python

.. image:: https://img.shields.io/pypi/v/pykrige.svg
    :target: https://pypi.python.org/pypi/pykrige

.. image:: https://anaconda.org/conda-forge/pykrige/badges/version.svg
  :target: https://github.com/conda-forge/pykrige-feedstock

.. image:: https://readthedocs.org/projects/pykrige/badge/?version=latest
    :target: http://pykrige.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/bsmurphy/PyKrige.svg?branch=master
    :target: https://travis-ci.org/bsmurphy/PyKrige

.. image:: https://ci.appveyor.com/api/projects/status/github/bsmurphy/PyKrige?branch=master&svg=true
    :target: https://ci.appveyor.com/project/bsmurphy/pykrige



The code supports 2D and 3D ordinary and universal kriging. Standard variogram models (linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used. The 2D universal kriging code currently supports regional-linear, point-logarithmic, and external drift terms, while the 3D universal kriging code supports a regional-linear drift term in all three spatial dimensions. Both universal kriging classes also support generic 'specified' and 'functional' drift capabilities. With the 'specified' drift capability, the user may manually specify the values of the drift(s) at each data point and all grid points. With the 'functional' drift capability, the user may provide callable function(s) of the spatial coordinates that define the drift(s). The package includes a module that contains functions that should be useful in working with ASCII grid files (`*.asc`).

See the `documentation <http://pykrige.readthedocs.io/en/latest/>`_ for more details.

Installation
^^^^^^^^^^^^

PyKrige requires Python 2.7 or 3.5+ as well as numpy, scipy and matplotlib. It can be installed from PyPi with,

.. code:: bash

    pip install pykrige

scikit-learn is an optional dependency needed for parameter tuning and regression kriging.


If you use conda, PyKrige can be installed from the `conda-forge` channel with,

.. code:: bash

    conda install -c conda-forge pykrige


Ordinary Kriging Example
^^^^^^^^^^^^^^^^^^^^^^^^

First we will create a 2D dataset together with the associated x, y grids,

.. code:: python

    import numpy as np
    import pykrige.kriging_tools as kt
    
    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])
    
    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)



Then we create the ordinary kriging object. The required inputs are the X-coordinates of
the data points, the Y-coordinates of the data points, and the Z-values of the
data points. If no variogram model is specified, defaults to a linear variogram
model. If no variogram model parameters are specified, then the code automatically
calculates the parameters by fitting the variogram model to the binned 
experimental semivariogram. The verbose kwarg controls code verbosity, and
the ``enable_plotting`` kwarg controls the display of the semivariogram.

    
.. code:: python

    from pykrige.ok import OrdinaryKriging

    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                         verbose=False, enable_plotting=False)
    					 
Next, we will create the kriged grid and the variance grid. Kriging on a rectangular
grid of points, on a masked rectangular grid of points, or with arbitrary points is allowed,

.. code:: python

    z, ss = OK.execute('grid', gridx, gridy)
    
Finally, we write the kriged grid to an ASCII grid file,

.. code:: python

    kt.write_asc_grid(gridx, gridy, z, filename="output.asc")


Other examples can be found in the `example gallery <http://pykrige.readthedocs.io/en/latest/examples/index.html>`_. 


Parameter Tuning
^^^^^^^^^^^^^^^^

A scikit-learn compatible API for parameter tuning by cross-validation is exposed in `sklearn.model_selection.GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_. See the `Krige CV <http://pykrige.readthedocs.io/en/latest/examples/krige_cv.html#sphx-glr-examples-krige-cv-py>`_ example for a more practical illustration.

In it's current form, the `pykrige.rk.Krige <http://pykrige.readthedocs.io/en/latest/generated/pykrige.rk.Krige.html#pykrige.rk.Krige>`_ class can be used to optimise all the common parameters of ``OrdinaryKriging`` and ``UniversalKriging``.


Regression Kriging
^^^^^^^^^^^^^^^^^^

`Regression kriging <https://en.wikipedia.org/wiki/Regression-Kriging>`_ can be performed with `pykrige.rk.RegressionKriging <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_. This class takes as parameters a scikit-learn regression model, and details of either either the ``OrdinaryKriging`` or the ``UniversalKriging`` class, and performs a correction steps on the ML regression prediction.
 
A demonstration of the regression kriging is provided in the `corresponding example <http://pykrige.readthedocs.io/en/latest/examples/regression_kriging2d.html#sphx-glr-examples-regression-kriging2d-py>`_.

License
^^^^^^^

PyKrige uses the BSD 3-Clause License.
