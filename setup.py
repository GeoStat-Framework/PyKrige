from setuptools import setup, Extension
from Cython.Distutils import build_ext
from os.path import join
import numpy as np

import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

ext_modules = [Extension("pykrige.lib.cok", ["pykrige/lib/cok.pyx"], extra_compile_args=['-O3', '-march=native']),
               Extension("pykrige.lib.lapack", ["pykrige/lib/lapack.pyx"], extra_compile_args=['-O2', '-march=native']),
               Extension("pykrige.lib.variogram_models", ["pykrige/lib/variogram_models.pyx"],
                         extra_compile_args=['-O2', '-march=native'])]

setup(name='PyKrige',
      version='1.2.0',
      author='Benjamin S. Murphy',
      author_email='bscott.murphy@gmail.com',
      url='https://github.com/bsmurphy/PyKrige',
      description='Kriging Toolkit for Python',
      long_description='PyKrige is a kriging toolkit for Python that supports \
      two- and three-dimensional ordinary and universal kriging.',
      packages=['pykrige'],
      package_data={'pykrige': ['README.md', 'CHANGELOG.md', 'LICENSE.txt', 'MANIFEST.in',
                    join('test_data', '*.txt'), join('test_data', '*.asc')]},
      requires=['numpy', 'scipy', 'matplotlib', 'Cython'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: GIS'],
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      cmdclass={'build_ext': build_ext},
      )
