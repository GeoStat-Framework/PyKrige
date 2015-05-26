PyKrige
=======

Kriging Toolkit for Python

The code supports two-dimensional ordinary and universal kriging as well as three-dimensional ordinary kriging. The universal kriging code currently supports regional-linear, point-logarithmic, and external drift terms. Standard variogram models (linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used with the code. Three-dimensional ordinary kriging, two-dimensional ordinary kriging, and two-dimensional universal kriging are separated into three classes. Examples of their uses are shown below. The code also includes a module that contains functions that should be useful in working with ASCII grid files (*.asc).

PyKrige is on PyPi, so installation is as simple as typing the following into a command line.
```shell
pip install pykrige
```

To update PyKrige from PyPi, type the following into a command line.
```shell
pip install --upgrade pykrige
```

PyKrige uses the BSD 3-Clause License.

Ordinary Kriging Example
------------------------

```python
from pykrige.ok import OrdinaryKriging
import numpy as np
import pykrige.kriging_tools as kt

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
```

Universal Kriging Example
-------------------------

```python
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
```

To Do...
--------
Someday the code will implement more drift terms, such as stream (line) and pond (polygon) drifts (for hydrological applications) and higher order polynomial drifts.