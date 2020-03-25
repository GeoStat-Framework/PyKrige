# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack
#from cpython.cobject cimport PyCObject_AsVoidPtr # only works with python 2, use capsules for python 3
from cpython.pycapsule cimport PyCapsule_GetPointer

cdef extern from "Python.h":
    void PyErr_Clear()


np.import_array()


# Snippet adapted from
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/src/capsule.h
cdef void* Capsule_AsVoidPtr(object obj):
    cdef void* ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL):
        PyErr_Clear()
    return ret


# interface to BLAS matrix-vector multipilcation though scipy.linalg 
# adapted from
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/kalman_loglike.pyx

cdef dgemv_t *dgemv = <dgemv_t*>Capsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemv', dtype='float64')._cpointer)
cdef dgesv_t *dgesv = <dgesv_t*>Capsule_AsVoidPtr(scipy.linalg.lapack.get_lapack_funcs('gesv', dtype='float64')._cpointer)
