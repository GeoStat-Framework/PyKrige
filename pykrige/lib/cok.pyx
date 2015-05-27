# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
#from scipy.linalg._fblas import dgemv # this is slower but more general
from libc.math cimport sqrt
import scipy.linalg
#from .lapack cimport dgemv
from .variogram_models cimport get_variogram_model


cpdef _c_exec_loop(double [:, ::1] a_all,
              double [:, ::1] bd_all,
              mask,
              long n,
              dict pars):
    cdef long i, j, k

    npt = bd_all.shape[0]

    cdef double [::1] zvalues = np.zeros(npt, dtype='float64')
    cdef double [::1] sigmasq = np.zeros(npt, dtype='float64')
    cdef double [::1] res = np.zeros(n+1, dtype='float64')
    cdef double [::1] tmp = np.zeros(n+1, dtype='float64')
    cdef double [::1] b = np.zeros(n+1, dtype='float64')
    cdef double [::1] Z = pars['Z']
    cdef double xpt, ypt
    cdef double z, ss
    cdef double eps=pars['eps']


    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])

    a_inv = scipy.linalg.inv(a_all)

    for i in range(npt):   # same thing as range(npt) if mask is not defined, otherwise take the non masked elements
        if mask[i]:
            continue
        bd = bd_all[i]
        if np.any(np.absolute(bd) <= eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= eps)
        else:
            zero_index = None
            zero_value = False
        c_variogram_function(variogram_model_parameters, bd, b)
        for k in range(n):
            b[k] *= -1


        #if zero_value:
        #    b[zero_index[0], 0] = 0.0
        b[n] = 1.0
        x = np.dot(a_inv, b)
        zvalues[i] = x[:n].dot(Z)
        sigmasq[i] = -x[:].dot(b[:])

    return zvalues.base, sigmasq.base

#cpdef _c_loop_old(double [:, ::1] a_all,
#              double [:, ::1] bd_all,
#              double [:,::1] masl,
#              long n,
#              dict pars):
#    cdef int np, n, i, j
#
#    npt = bd_all.shape[0]
#    a_inv = scipy.linalg.inv(a)
#
#    cdef double [:,::1] gridz = np.zeros((ny, nx), dtype='float64')
#    cdef double [:,::1] sigmasq = np.zeros((ny, nx), dtype='float64')
#    cdef double [::1] res = np.zeros(b.shape[0], dtype='float64')
#    cdef double [::1] tmp = np.zeros(n, dtype='float64')
#    cdef double xpt, ypt
#    cdef double z, ss
#
#
#    c_variogram_function = get_variogram_model(pars['variogram_function'].__name__)
#
#    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])
#
#    for i in range(npt):
#            xpt = grid_x[i, j]
#            ypt = grid_y[i, j]
#
#            _apply_dot_product(x_adjusted, y_adjusted, z_in,
#                                            xpt, ypt,
#                                            bd, res, tmp,
#                                            c_variogram_function, variogram_model_parameters,
#                                            Ai, b, pars, &z, &ss)
#            gridz[i, j] = z
#            sigmasq[i, j] = ss
#    return gridz, sigmasq
#
#
#cdef int _apply_dot_product(double [::1] x, double [::1] y, double[::1] z,
#                            float xpt, float ypt,
#                            double [::1] bd, double [::1] res, double [::1] tmp,
#                            variogram_model_t variogram_function, double [::1] variogram_model_parameters,
#                            double [:,:] Ai, double [::1] b, dict pars,
#                            double * zinterp, double *sigmasq):
#    cdef int n_withdrifts, n, index, k, l, nb, inc=1
#    cdef double alpha=1.0, beta=0.0
#    n_withdrifts = pars['n_withdrifts']
#    n = x.shape[0]
#    nb = b.shape[0]
#
#    for k in range(n):
#        bd[k] = sqrt((x[k] - xpt)*(x[k] - xpt) + (y[k] - ypt)*(y[k] - ypt))
#
#    variogram_function(variogram_model_parameters, bd, n,  tmp)
#    for k in range(n):
#        b[k] = -tmp[k]
#    index = n
#    if pars['regional_linear_drift']:
#        b[index] = xpt
#        index += 1
#        b[index] = ypt
#        index += 1
#    if pars['point_log_drift']:
#        point_log_array = pars['point_log_array']
#        for well_no in range(point_log_array.shape[0]):
#            dist = np.sqrt((xpt - point_log_array[well_no, 0])**2 +
#                           (ypt - point_log_array[well_no, 1])**2)
#            b[index] = - point_log_array[well_no, 2] * np.log(dist)
#            index += 1
#
#    if pars['external_Z_drift']:
#        b[index] = pars['_calculate_data_point_zscalars'](np.array([xpt]),
#                                                          np.array([ypt]))
#        index += 1
#
#    if index != n_withdrifts:
#        print("WARNING: Error in setting up kriging system. Kriging may fail.")
#
#    if pars['UNBIAS']:
#        b[n_withdrifts] = 1.0
#
#    #dgemv(1.0, Ai, b, y=res, overwrite_y=1) # alternative formulation
#    dgemv(
#                              # # Compute y := alpha*A*x + beta*y
#         'N',                 # char *trans, # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
#         &nb,                 # int *m, # Rows of A (prior to transpose from *trans)
#         &nb,                 # int *n, # Columns of A / min(len(x))
#         &alpha,              # np.float64_t *alpha, # Scalar multiple
#         &(Ai[0,0]),          # np.float64_t *a, # Matrix A: mxn
#         &nb,                 # int *lda, # The size of the first dimension of A (in memory)
#         &(b[0]),             # np.float64_t *x, # Vector x, min(len(x)) = n
#         &inc,                # int *incx, # The increment between elements of x (usually 1)
#         &beta,               # np.float64_t *beta, # Scalar multiple
#         &(res[0]),           # np.float64_t *y, # Vector y, min(len(y)) = m
#         &inc                 # int *incy # The increment between elements of y (usually 1)
#        )
#
#    zinterp[0] = 0.0
#    sigmasq[0] = 0.0
#    for k in range(n):
#        zinterp[0] += res[k]*z[k]
#    for k in range(nb):
#        sigmasq[0] += res[k]*(-b[k])
#
#    return 0


