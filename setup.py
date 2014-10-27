from distutils.core import setup
from os.path import join

setup(name='PyKrige',
      version='0.1',
      author='Benjamin S. Murphy',
      author_email='bscott.murphy@gmail.com',
      url='https://github.com/bsmurphy/PyKrige',
      description='Kriging Toolkit for Python',
      long_description='PyKrige is a kriging toolkit for Python that currently supports \
      two-dimensional ordinary and universal kriging. Regional-linear, point-logarithmic, \
      and external-scalar drift terms are currently supported.',
      packages=['pykrige'],
      # package_data={'pykrige':['README.md', 'LICENSE.txt', \
      # 	          join('test', '*.txt'), join('test', '*.asc')]},
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: GIS']
      )
