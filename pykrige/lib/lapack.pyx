# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack
from cpython.cobject cimport PyCObject_AsVoidPtr # only works with python 2, use capsules for python 3

np.import_array()


# interface to BLAS matrix-vector multipilcation though scipy.linalg 
# adapted from
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/kalman_loglike.pyx


cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemv', dtype='float64')._cpointer)
cdef dgesv_t *dgesv = <dgesv_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.get_lapack_funcs('gesv', dtype='float64')._cpointer)
