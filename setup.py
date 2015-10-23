from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from os.path import join
import numpy as np

import Cython.Compiler.Options

parser = argparse.ArgumentParser()
parser.add_argument("--no_cython", help="Disable compilation of Cython extensions.", action='store_true')
args = parser.parse_args()

Cython.Compiler.Options.annotate = False


class BuildExtCompilerCheck(build_ext):
    def build_extensions(self):
        if sys.platform == 'win32' and ('MSC' in sys.version or 'MSVC' in sys.version):
            print("COMPILER IS", self.compiler.compiler_type)
            from distutils.msvccompiler import MSVCCompiler
            if isinstance(self.compiler, MSVCCompiler):
                build_ext.build_extensions(self)
            else:
                print("WARNING: The C extensions will not be built since the necessary compiler could not be found.\n"
                      "See https://github.com/bsmurphy/PyKrige/issues/8")
        else:
            build_ext.build_extensions(self)


if args.no_cython:
    print("DISABLING CYTHON EXTENSIONS.")
    ext_modules = []
    cmd = {}
elif sys.version[0] == '3':
    print("WARNING: Currently, Cython extensions are not built when using Python 3. "
          "This will be changed in the future.")
    ext_modules = []
    cmd = {}
else:
    if sys.platform != 'win32':
        compile_args = dict(extra_compile_args=['-O2', '-march=core2', '-mtune=corei7'],
                            extra_link_args=['-O2', '-march=core2', '-mtune=corei7'])
    else:
        compile_args = {}
    ext_modules = [Extension("pykrige.lib.cok", ["pykrige/lib/cok.pyx"], **compile_args),
                   Extension("pykrige.lib.lapack", ["pykrige/lib/lapack.pyx"], **compile_args),
                   Extension("pykrige.lib.variogram_models", ["pykrige/lib/variogram_models.pyx"], **compile_args)]
    cmd = {'build_ext': BuildExtCompilerCheck}

setup(name='PyKrige',
      version='1.3.0',
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
      cmdclass=cmd)