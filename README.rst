PyKrige
=======

.. image:: https://badge.fury.io/py/PyKrige.svg
   :target: https://badge.fury.io/py/PyKrige
.. image:: https://img.shields.io/conda/vn/conda-forge/pykrige.svg
   :target: https://anaconda.org/conda-forge/pykrige
.. image:: https://readthedocs.org/projects/pykrige/badge/?version=latest
   :target: http://pykrige.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://travis-ci.com/GeoStat-Framework/PyKrige.svg?branch=master
   :target: https://travis-ci.com/GeoStat-Framework/PyKrige


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

See the documentation at `http://pykrige.readthedocs.io/ <http://pykrige.readthedocs.io/>`_ for more details.

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


Ordinary Kriging Example
^^^^^^^^^^^^^^^^^^^^^^^^

First we will create a 2D dataset together with the associated x, y grids,

.. code:: python

    import numpy as np
    import pykrige.kriging_tools as kt
    from pykrige.ok import OrdinaryKriging

    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])

    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)

    # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values of the
    # data points. If no variogram model is specified, defaults to a linear variogram
    # model. If no variogram model parameters are specified, then the code automatically
    # calculates the parameters by fitting the variogram model to the binned
    # experimental semivariogram. The verbose kwarg controls code talk-back, and
    # the enable_plotting kwarg controls the display of the semivariogram.
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                         verbose=False, enable_plotting=False)

    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # (See OrdinaryKriging.__doc__ for more information.)
    z, ss = OK.execute('grid', gridx, gridy)

    # Writes the kriged grid to an ASCII grid file.
    kt.write_asc_grid(gridx, gridy, z, filename="output.asc")


Universal Kriging Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pykrige.uk import UniversalKriging
    import numpy as np

    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])

    gridx = np.arange(0.0, 5.5, 0.5)
    gridy = np.arange(0.0, 5.5, 0.5)

    # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # the data points, the Y-coordinates of the data points, and the Z-values of the
    # data points. Variogram is handled as in the ordinary kriging case.
    # drift_terms is a list of the drift terms to include; currently supported terms
    # are 'regional_linear', 'point_log', and 'external_Z'. Refer to
    # UniversalKriging.__doc__ for more information.
    UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                          drift_terms=['regional_linear'])

    # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # (See UniversalKriging.__doc__ for more information.)
    z, ss = UK.execute('grid', gridx, gridy)


Three-Dimensional Kriging Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pykrige.ok3d import OrdinaryKriging3D
    from pykrige.uk3d import UniversalKriging3D
    import numpy as np

    data = np.array([[0.1, 0.1, 0.3, 0.9],
                                     [0.2, 0.1, 0.4, 0.8],
                                     [0.1, 0.3, 0.1, 0.9],
                                     [0.5, 0.4, 0.4, 0.5],
                                     [0.3, 0.3, 0.2, 0.7]])

    gridx = np.arange(0.0, 0.6, 0.05)
    gridy = np.arange(0.0, 0.6, 0.01)
    gridz = np.arange(0.0, 0.6, 0.1)

    # Create the 3D ordinary kriging object and solves for the three-dimension kriged
    # volume and variance. Refer to OrdinaryKriging3D.__doc__ for more information.
    ok3d = OrdinaryKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                                     variogram_model='linear')
    k3d, ss3d = ok3d.execute('grid', gridx, gridy, gridz)

    # Create the 3D universal kriging object and solves for the three-dimension kriged
    # volume and variance. Refer to UniversalKriging3D.__doc__ for more information.
    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                                      variogram_model='linear', drift_terms=['regional_linear'])
    k3d, ss3d = uk3d.execute('grid', gridx, gridy, gridz)

    # To use the generic 'specified' drift term, the user must provide the drift values
    # at each data point and at every grid point. The following example is equivalent to
    # using a linear drift in all three spatial dimensions. Refer to
    # UniversalKriging3D.__doc__ for more information.
    zg, yg, xg = np.meshgrid(gridz, gridy, gridx, indexing='ij')
    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                                      variogram_model='linear', drift_terms=['specified'],
                                                      specified_drift=[data[:, 0], data[:, 1]])
    k3d, ss3d = uk3d.execute('grid', gridx, gridy, gridz, specified_drift_arrays=[xg, yg, zg])

    # To use the generic 'functional' drift term, the user must provide a callable
    # function that takes only the spatial dimensions as arguments. The following example
    # is equivalent to using a linear drift only in the x-direction. Refer to
    # UniversalKriging3D.__doc__ for more information.
    func = lambda x, y, z: x
    uk3d = UniversalKriging3D(data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                                                      variogram_model='linear', drift_terms=['functional'],
                                                      functional_drift=[func])
    k3d, ss3d = uk3d.execute('grid', gridx, gridy, gridz)

    # Note that the use of the 'specified' and 'functional' generic drift capabilities is
    # essentially identical in the two-dimensional universal kriging class (except for a
    # difference in the number of spatial coordinates for the passed drift functions).
    # See UniversalKriging.__doc__ for more information.


GSTools covariance models
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use `GSTools <https://github.com/GeoStat-Framework/GSTools>`_
covariance models as input for the ``variogram_model`` in the
PyKrige routines:

.. code:: python

    import numpy as np
    from gstools import Gaussian
    from pykrige.ok import OrdinaryKriging
    from matplotlib import pyplot as plt

    # conditioning data
    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])
    # grid definition for output field
    gridx = np.arange(0.0, 5.5, 0.1)
    gridy = np.arange(0.0, 6.5, 0.1)
    # a GSTools based covariance model
    cov_model = Gaussian(dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
    # ordinary kriging with pykrige
    OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], cov_model)
    z1, ss1 = OK1.execute('grid', gridx, gridy)
    plt.imshow(z1, origin="lower")
    plt.show()

Which gives:

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/GSTools/master/docs/source/pics/20_pykrige.png
   :width: 400px
   :align: center

Have a look at the `documentation about the Covariance Model of GSTools <https://geostat-framework.readthedocs.io/projects/gstools/en/latest/tutorial_02_cov.html>`_.


Kriging Parameters Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^

A scikit-learn compatible API for parameter tuning by cross-validation is exposed in
`sklearn.model_selection.GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
See the `Krige CV <http://pykrige.readthedocs.io/en/latest/examples/krige_cv.html#sphx-glr-examples-krige-cv-py>`_
example for a more practical illustration.


Regression Kriging
^^^^^^^^^^^^^^^^^^

`Regression kriging <https://en.wikipedia.org/wiki/Regression-Kriging>`_ can be performed
with `pykrige.rk.RegressionKriging <http://pykrige.readthedocs.io/en/latest/examples/regression_kriging2d.html>`_.
This class takes as parameters a scikit-learn regression model, and details of either either
the ``OrdinaryKriging`` or the ``UniversalKriging`` class, and performs a correction steps on the ML regression prediction.

A demonstration of the regression kriging is provided in the
`corresponding example <http://pykrige.readthedocs.io/en/latest/examples/regression_kriging2d.html#sphx-glr-examples-regression-kriging2d-py>`_.

License
^^^^^^^

PyKrige uses the BSD 3-Clause License.
