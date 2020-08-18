#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import scipy.linalg
from scipy.linalg.cython_blas cimport dgemv
from scipy.linalg.cython_lapack cimport dgesv
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

    nb = n + 1


    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])

    cdef double [::1,:] a_inv = np.asfortranarray(np.empty_like(a_all))


    if pars['pseudo_inv']:
        if pars['pseudo_inv_type'] == "pinv":
            a_inv = np.asfortranarray(scipy.linalg.pinv(a_all))
        elif pars['pseudo_inv_type'] == "pinv2":
            a_inv = np.asfortranarray(scipy.linalg.pinv2(a_all))
        elif pars['pseudo_inv_type'] == "pinvh":
            a_inv = np.asfortranarray(scipy.linalg.pinvh(a_all))
        else:
            raise ValueError('Unknown pseudo inverse method selected.')
    else:
        a_inv = np.asfortranarray(scipy.linalg.inv(a_all))


    for i in range(npt):   # same thing as range(npt) if mask is not defined, otherwise take the non masked elements
        if mask[i]:
            continue
        bd = bd_all[i]

        c_variogram_function(variogram_model_parameters, n, bd, b)

        for k in range(n):
            b[k] *= -1
        b[n] = 1.0

        if not pars['exact_values']:
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

cpdef _c_exec_loop_moving_window(double [:, ::1] a_all,
              double [:, ::1] bd_all,
              char [::1] mask,
              long [:,::1] bd_idx,
              long n_max,
              dict pars):
    cdef long i, j, k, p_i, p_j, npt, n

    npt = bd_all.shape[0]
    n = bd_idx.shape[1]

    cdef double [::1] zvalues = np.zeros(npt, dtype='float64')
    cdef double [::1] sigmasq = np.zeros(npt, dtype='float64')
    cdef double [::1] x = np.zeros(n_max+1, dtype='float64')
    cdef double [::1] tmp = np.zeros(n_max+1, dtype='float64')
    cdef double [::1] b = np.zeros(n_max+1, dtype='float64')
    cdef double [::1] Z = pars['Z']
    cdef double [::1] a_selection = np.zeros((n_max+1)*(n_max+1), dtype='float64')
    cdef double [::1] bd = np.zeros(n_max, dtype='float64')
    cdef double z_tmp, ss_tmp, eps=pars['eps']
    cdef int nb, nrhs=1, info
    cdef int [::1] ipiv = np.zeros(n_max+1, dtype='int32')
    cdef double alpha=1.0, beta=0.0
    cdef long [::1] bd_idx_sel

    nb = n + 1



    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])



    for i in range(npt):
        if mask[i]:
            continue

        bd = bd_all[i, :]

        bd_idx_sel = bd_idx[i, :]


        for k in range(n):
            p_i = bd_idx_sel[k]
            for j in range(n):
                p_j = bd_idx_sel[j]
                a_selection[k + nb*j] = a_all[p_i, p_j]

        for k in range(nb):
            a_selection[k*nb + n] = 1.0
            a_selection[n*nb + k] = 1.0
        a_selection[n*nb + n] = 0.0

        c_variogram_function(variogram_model_parameters, n, bd, tmp)

        for k in range(n):
            b[k] = - tmp[k]
        b[n] = 1.0

        check_b_vect(n, bd, b, eps)

        for k in range(nb):
            x[k] = b[k]

        # higher level (and slower) call to do the same thing as dgesv below
        #a2D = a_selection.base[:nb*nb].reshape((nb, nb))
        #x =  scipy.linalg.solve(a2D, b.base[:nb])

        dgesv(
                &nb,
                &nrhs,
                &(a_selection[0]),
                &nb,
                &(ipiv[0]),
                &(x[0]),
                &nb,
                &info
                )

        if info > 0:
            raise ValueError('Singular matrix')
        elif info < 0:
            raise ValueError('Wrong arguments')


        z_tmp = 0.0
        ss_tmp = 0.0
        for k in range(n):
            j = bd_idx_sel[k]
            z_tmp += x[k]*Z[j]

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
