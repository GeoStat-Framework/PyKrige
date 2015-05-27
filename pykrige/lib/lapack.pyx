# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
from cpython.cobject cimport PyCObject_AsVoidPtr # only works with python 2, use capsules for python 3

np.import_array()


# interface to BLAS matrix-vector multipilcation though scipy.linalg 
# adapted from
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/kalman_loglike.pyx

ctypedef int dgemv_t(
        # Compute y := alpha*A*x + beta*y
        char *trans, # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
        int *m, # Rows of A (prior to transpose from *trans)
        int *n, # Columns of A / min(len(x))
        np.float64_t *alpha, # Scalar multiple
        np.float64_t *a, # Matrix A: mxn
        int *lda, # The size of the first dimension of A (in memory)
        np.float64_t *x, # Vector x, min(len(x)) = n
        int *incx, # The increment between elements of x (usually 1)
        np.float64_t *beta, # Scalar multiple
        np.float64_t *y, # Vector y, min(len(y)) = m
        int *incy # The increment between elements of y (usually 1)
        )

cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemv', dtype='float64')._cpointer)
