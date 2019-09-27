# -*- coding: utf-8 -*-
"""
Example how to use the PyKrige routines with a GSTools CovModel
"""
import numpy as np
from gstools import Gaussian
from pykrige.ok import OrdinaryKriging
from matplotlib import pyplot as plt

data = np.array([[0.3, 1.2, 0.47],
                 [1.9, 0.6, 0.56],
                 [1.1, 3.2, 0.74],
                 [3.3, 4.4, 1.47],
                 [4.7, 3.8, 1.74]])

gridx = np.arange(0.0, 5.5, 0.1)
gridy = np.arange(0.0, 6.5, 0.1)

# a GSTools based covariance model
model = Gaussian(dim=2, len_scale=1, anis=0.2, angles=0.5, var=0.5, nugget=0.1)

OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], model)
z1, ss1 = OK1.execute('grid', gridx, gridy)
plt.imshow(z1, origin="lower")
plt.show()
