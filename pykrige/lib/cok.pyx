# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
from libc.math cimport sqrt
import scipy.linalg
from .lapack cimport dgemv, dgesv
from .variogram_models cimport get_variogram_model


cpdef _c_exec_loop(double [:, ::1] a_all,
              double [:, ::1] bd_all,
              char [::1] mask,
              long n,
              dict pars):
    cdef long i, j, k

    npt = bd_all.shape[0]

    cdef double [::1] zvalues = np.zeros(npt, dtype='float64')
    cdef double [::1] sigmasq = np.zeros(npt, dtype='float64')
    cdef double [::1] x = np.zeros(n+1, dtype='float64')
    cdef double [::1] b = np.zeros(n+1, dtype='float64')
    cdef double [::1] Z = pars['Z']
    cdef double [::1] bd
    cdef double z_tmp, ss_tmp, eps=pars['eps']
    cdef int nb, inc=1
    cdef double alpha=1.0, beta=0.0

    nb = n +1


    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])

    cdef double [::1,:] a_inv = scipy.linalg.inv(a_all)

    for i in range(npt):   # same thing as range(npt) if mask is not defined, otherwise take the non masked elements
        if mask[i]:
            continue
        bd = bd_all[i]

        c_variogram_function(variogram_model_parameters, n, bd, b)

        for k in range(n):
            b[k] *= -1
        b[n] = 1.0

        check_b_vect(n, bd, b, eps)


        # Do the BLAS matrix-vector multiplication call
        dgemv(
                                  # # Compute y := alpha*A*x + beta*y
             'N',                 # char *trans, # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
             &nb,                 # int *m, # Rows of A (prior to transpose from *trans)
             &nb,                 # int *n, # Columns of A / min(len(x))
             &alpha,              # np.float64_t *alpha, # Scalar multiple
             &(a_inv[0,0]),          # np.float64_t *a, # Matrix A: mxn
             &nb,                 # int *lda, # The size of the first dimension of A (in memory)
             &(b[0]),             # np.float64_t *x, # Vector x, min(len(x)) = n
             &inc,                # int *incx, # The increment between elements of x (usually 1)
             &beta,               # np.float64_t *beta, # Scalar multiple
             &(x[0]),           # np.float64_t *y, # Vector y, min(len(y)) = m
             &inc                 # int *incy # The increment between elements of y (usually 1)
            )

        z_tmp = 0.0
        ss_tmp = 0.0
        for k in range(n):
            z_tmp += x[k]*Z[k]

        for k in range(nb):
            ss_tmp += x[k]*b[k]

        zvalues[i] = z_tmp
        sigmasq[i] = -ss_tmp

    return zvalues.base, sigmasq.base

cpdef _c_exec_loop_mooving_window(double [:, ::1] a_all,
              double [:, ::1] bd_all,
              char [::1] mask,
              long [:,:] bd_idx,
              long n_max,
              dict pars):
    cdef long i, j, k, p, j_2, npt, n

    npt = bd_all.shape[0]
    n = bd_idx.shape[1]

    cdef double [::1] zvalues = np.zeros(npt, dtype='float64')
    cdef double [::1] sigmasq = np.zeros(npt, dtype='float64')
    cdef double [::1] x = np.zeros(n_max+1, dtype='float64')
    cdef double [::1] b = np.zeros(n_max+1, dtype='float64')
    cdef double [::1] Z = pars['Z']
    cdef double [::1] a_selection = np.zeros((n_max+1)*(n_max+1), dtype='float64')
    cdef double [::1] bd = np.zeros(n_max, dtype='float64')
    cdef double z_tmp, ss_tmp, eps=pars['eps']
    cdef int nb, nrhs=1, info, n_int
    cdef int [::1] ipiv = np.zeros(n_max+1, dtype='int32')
    cdef double alpha=1.0, beta=0.0

    nb = n +1

    n_int = n+1


    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])


    return zvalues.base, sigmasq.base
    for i in range(npt):   # same thing as range(npt) if mask is not defined, otherwise take the non masked elements
        if mask[i]:
            continue

        for k in range(n):
            bd[k] = bd_all[i, k]

        for k in range(n):
            j = bd_idx[i, k]
            for p in range(n):
                j_2 = bd_idx[i, p]
                a_selection[k*n_int + p] = a_all[j, j_2]

        for k in range(n+1):
            a_selection[k*n_int + n] = 1.0
            a_selection[n*n_int + k] = 1.0
        a_selection[n*n_int + n] = 0.0

        c_variogram_function(variogram_model_parameters, n, bd, b)

        for k in range(n):
            b[k] *= -1
        b[n] = 1.0

        check_b_vect(n, bd, b, eps)

        for k in range(n+1):
            x[k] = b[k]

        #a2D = a_selection.base[:n_int*n_int].reshape((n_int, n_int))
        #x =  scipy.linalg.solve(a2D, b.base[:n+1])

        dgesv(
                &n_int,
                &nrhs,
                &(a_selection[0]),
                &n_int,
                &(ipiv[0]),
                &(x[0]),
                &n_int,
                &info
                )

        if info > 0:
            raise ValueError('Syngular matrix')
        elif info < 0:
            raise ValueError('Wrong arguments')


        z_tmp = 0.0
        ss_tmp = 0.0
        for k in range(n):
            z_tmp += x[k]*Z[k]

        for k in range(nb):
            ss_tmp += x[k]*b[k]

        zvalues[i] = z_tmp
        sigmasq[i] = -ss_tmp

    return zvalues.base, sigmasq.base


cdef int check_b_vect(long n, double [::1] bd, double[::1] b, double eps) nogil:
    cdef long k

    for k in range(n):
        if bd[k] <= eps:
            b[k] = 0.0

    return 0
