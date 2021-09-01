# -*- coding: utf-8 -*-
"""Kriging Toolkit for Python."""
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# cython extensions ###########################################################

CY_MODULES = []
CY_MODULES.append(
    Extension(
        "pykrige.lib.cok",
        [os.path.join("pykrige", "lib", "cok.pyx")],
        include_dirs=[np.get_include()],
    )
)
CY_MODULES.append(
    Extension(
        "pykrige.lib.variogram_models",
        [os.path.join("pykrige", "lib", "variogram_models.pyx")],
        include_dirs=[np.get_include()],
    )
)
EXT_MODULES = cythonize(CY_MODULES)  # annotate=True

# embed signatures for sphinx
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}

# setup #######################################################################

setup(ext_modules=EXT_MODULES, include_dirs=[np.get_include()])
