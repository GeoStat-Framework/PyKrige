from setuptools import setup, Extension
from Cython.Distutils import build_ext
from os.path import join
import numpy as np

import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

ext_modules=[
        Extension("pykrige.lib.cuk",
                ["pykrige/lib/cuk.pyx"],
                extra_compile_args=['-g', '-O3', '-march=native'],
                extra_link_args=['-g', '-O3', '-march=native']
                )]

setup(name='PyKrige',
      version='1.0.3',
      author='Benjamin S. Murphy',
      author_email='bscott.murphy@gmail.com',
      url='https://github.com/bsmurphy/PyKrige',
      description='Kriging Toolkit for Python',
      long_description='PyKrige is a kriging toolkit for Python that currently supports \
      two-dimensional ordinary and universal kriging. Regional-linear, point-logarithmic, \
      and external-scalar drift terms are currently supported.',
      packages=['pykrige'],
      package_data={'pykrige':['README.md', 'CHANGELOG.md', 'LICENSE.txt', 'MANIFEST.in',
                    join('test_data', '*.txt'), join('test_data', '*.asc')]},
      requires=['numpy', 'scipy', 'matplotlib'],
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
