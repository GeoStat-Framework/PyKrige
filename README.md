PyKrige
=======

Kriging Toolkit for Python

Currently, the code supports two-dimensional ordinary and universal kriging. The universal kriging code currently supports regional-linear, point-logarithmic, and external drift terms. Ordinary and universal kriging are separated into two classes. Examples of their uses are shown below. The code also includes a module that contains functions that should be useful in working with ASCII grid files (*.asc).

Ordinary Kriging Example
------------------------

```python
from pykrige.ok import OrdinaryKriging
import numpy as np
import pykrige.kriging_tools as kt

data = np.array([[2.3, 4.5, 0.3],
                 [4.7, 1.4, 0.6],
                 [8.2, 7.4, 1.6],
                 [3.6, 8.5, 1.3],
                 [1.2, 7.6, 4.7]])

gridx = np.arange(0.0, 10.0, 0.5)
gridy = np.arange(0.0, 10.0, 0.5)

# Create the ordinary kriging object. Required inputs are the X-coordinates of
# the data points, the Y-coordinates of the data points, and the Z-values of the
# data points. If no variogram model is specified, defaults to a linear variogram
# model. If no variogram model parameters are specified, then the code automatically
# calculates the parameters by fitting the variogram model to the binned 
# experimental semivariogram. The verbose kwarg controls code talk-back, and
# the enable_plotting kwarg controls the display of the semivariogram.
OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                     verbose=False, enable_plotting=False)
					 
# Creates the kriged grid and the variance grid.
z, ss = OK.execute(gridx, gridy)

# Writes the kriged grid to an ASCII grid file.
kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
```

Universal Kriging Example
-------------------------

```python
from pykrige.uk import UniversalKriging
import numpy as np

data = np.array([[2.3, 4.5, 0.3],
                 [4.7, 1.4, 0.6],
                 [8.2, 7.4, 1.6],
                 [3.6, 8.5, 1.3],
                 [1.2, 7.6, 4.7]])

gridx = np.arange(0.0, 10.0, 0.5)
gridy = np.arange(0.0, 10.0, 0.5)

# Create the ordinary kriging object. Required inputs are the X-coordinates of
# the data points, the Y-coordinates of the data points, and the Z-values of the
# data points. Variogram is handled as in the ordinary kriging case.
# drift_terms is a list of the drift terms to include; currently supported terms
# are 'regional_linear', 'point_log', and 'external_Z'. Refer to 
# UniversalKriging.__doc__ for more information.
UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                      drift_terms=['regional_linear'])
					 
# Creates the kriged grid and the variance grid.
z, ss = UK.execute(gridx, gridy)
```

To Do...
--------
- Consolidate core kriging functions to reduce redundancy in code.
