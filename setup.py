# -*- coding: utf-8 -*-
"""Kriging Toolkit for Python."""
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# cython extensions
CY_MODULES = [
    Extension(
        name=f"pykrige.{ext}",
        sources=[os.path.join("src", "pykrige", *ext.split(".")) + ".pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for ext in ["lib.cok", "lib.variogram_models"]
]

# setup - do not include package data to ignore .pyx files in wheels
setup(ext_modules=cythonize(CY_MODULES), include_package_data=False)
