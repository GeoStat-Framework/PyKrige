PyKrige
=======

Kriging Toolkit for Python

[![Build Status](https://travis-ci.org/bsmurphy/PyKrige.svg?branch=master)](https://travis-ci.org/bsmurphy/PyKrige)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/bsmurphy/PyKrige?branch=master&svg=true)](https://ci.appveyor.com/project/bsmurphy/pykrige)

The code supports two- and three- dimensional ordinary and universal kriging. Standard variogram models (linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used with the code. The kriging methods are separated into four classes. Examples of their uses are shown below. The two-dimensional universal kriging code currently supports regional-linear, point-logarithmic, and external drift terms, while the three-dimensional universal kriging code supports a regional-linear drift term in all three spatial dimensions. Both universal kriging classes also support generic 'specified' and 'functional' drift capabilities. With the 'specified' drift capability, the user may manually specify the values of the drift(s) at each data point and all grid points. With the 'functional' drift capability, the user may provide callable function(s) of the spatial coordinates that define the drift(s). The package includes a module that contains functions that should be useful in working with ASCII grid files (*.asc).

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

Three-Dimensional Kriging Example
-------------------------

```python
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
```
