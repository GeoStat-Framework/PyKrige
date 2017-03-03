#!/bin/python
from pykrige.ok import OrdinaryKriging
import numpy as np

# Make this example reproducible:
np.random.seed(89239413)

# Generate random data following a uniform spatial distribution
# of nodes and a uniform distribution of values in the interval
# [2.0, 5.5]:
N = 10
lon = 360.0*np.random.random(N)
lat = 180.0/np.pi*np.arcsin(2*np.random.random(N)-1)
z   = 3.5*np.random.rand(N, 1) + 2.0

# Generate a regular grid with 60° longitude and 30° latitude steps:
grid_lon = np.linspace(0.0, 360.0, 7)
grid_lat = np.linspace(-90.0, 90.0, 7)

# Create ordinary kriging object:
OK = OrdinaryKriging(lon, lat, z, variogram_model='linear', verbose=False,
                     enable_plotting=False, coordinates_type='geographic')

# Execute on grid:
z, ss = OK.execute('grid', grid_lon, grid_lat)

# Print data at equator (last longitude index will show periodicity):
print("Data at equator:\n================")
print("Longitude:",grid_lon)
print("Value:    ",np.array_str(z[3,:], precision=2))
print("Sigma²:   ",np.array_str(ss[3,:], precision=2))

##====================================OUTPUT==================================
# Data at equator:
# ================
# Longitude: [   0.   60.  120.  180.  240.  300.  360.]
# Value:     [ 3.67  3.42  3.33  3.94  4.19  4.28  3.67]
# Sigma²:    [ 1.59  1.08  1.04  1.31  1.86  1.99  1.59]
