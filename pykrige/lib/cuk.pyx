# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
import scipy.linalg.blas
#from scipy.linalg._fblas import dgemv # this is slower but more general
from libc.math cimport sqrt
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

#ctypedef void (*variogram_model_t)(double [::1], np.ndarray[np.float64_t], np.ndarray[np.float64_t])
ctypedef void (*variogram_model_t)(double [::1], double [::1], int, double [::1])


cpdef _c_loop(double [:, ::1] grid_x, double [:, ::1] grid_y,
              double [::1] x_adjusted, double [::1] y_adjusted,
              double [::1] z_in,
              double [:,:] Ai,
              double [::1] b,
              dict pars):
    cdef int nx, ny, n, i, j

    nx = grid_x.shape[0]
    ny = grid_x.shape[1]
    n = x_adjusted.shape[0]

    cdef double [:,::1] gridz = np.zeros((ny, nx), dtype='float64')
    cdef double [:,::1] sigmasq = np.zeros((ny, nx), dtype='float64')
    cdef double [::1] bd = np.zeros(n, dtype='float64')
    cdef double [::1] res = np.zeros(b.shape[0], dtype='float64')
    cdef double [::1] tmp = np.zeros(n, dtype='float64')
    cdef double xpt, ypt
    cdef double z, ss

    func_name = pars['variogram_function'].__name__

    cdef variogram_model_t c_variogram_function

    if func_name == 'linear_variogram_model':
        c_variogram_function = &_c_linear_variogram_model
    else:
        raise NotImplementedError

    cdef double [::1] variogram_model_parameters = np.asarray(pars['variogram_model_parameters'])


    for i in range(ny):
        for j in range(nx):
            xpt = grid_x[i, j]
            ypt = grid_y[i, j]

            _apply_dot_product(x_adjusted, y_adjusted, z_in,
                                            xpt, ypt,
                                            bd, res, tmp,
                                            c_variogram_function, variogram_model_parameters,
                                            Ai, b, pars, &z, &ss)
            gridz[i, j] = z
            sigmasq[i, j] = ss
    return gridz, sigmasq


cdef int _apply_dot_product(double [::1] x, double [::1] y, double[::1] z,
                            float xpt, float ypt,
                            double [::1] bd, double [::1] res, double [::1] tmp,
                            variogram_model_t variogram_function, double [::1] variogram_model_parameters,
                            double [:,:] Ai, double [::1] b, dict pars,
                            double * zinterp, double *sigmasq):
    cdef int n_withdrifts, n, index, k, l, nb, inc=1
    cdef double alpha=1.0, beta=0.0
    n_withdrifts = pars['n_withdrifts']
    n = x.shape[0]
    nb = b.shape[0]

    for k in range(n):
        bd[k] = sqrt((x[k] - xpt)*(x[k] - xpt) + (y[k] - ypt)*(y[k] - ypt))

    variogram_function(variogram_model_parameters, bd, n,  tmp)
    for k in range(n):
        b[k] = -tmp[k]
    index = n
    if pars['regional_linear_drift']:
        b[index] = xpt
        index += 1
        b[index] = ypt
        index += 1
    if pars['point_log_drift']:
        point_log_array = pars['point_log_array']
        for well_no in range(point_log_array.shape[0]):
            dist = np.sqrt((xpt - point_log_array[well_no, 0])**2 +
                           (ypt - point_log_array[well_no, 1])**2)
            b[index] = - point_log_array[well_no, 2] * np.log(dist)
            index += 1

    if pars['external_Z_drift']:
        b[index] = pars['_calculate_data_point_zscalars'](np.array([xpt]),
                                                          np.array([ypt]))
        index += 1

    if index != n_withdrifts:
        print("WARNING: Error in setting up kriging system. Kriging may fail.")

    if pars['UNBIAS']:
        b[n_withdrifts] = 1.0

    #dgemv(1.0, Ai, b, y=res, overwrite_y=1) # alternative formulation
    dgemv(
                              # # Compute y := alpha*A*x + beta*y
         'N',                 # char *trans, # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
         &nb,                 # int *m, # Rows of A (prior to transpose from *trans)
         &nb,                 # int *n, # Columns of A / min(len(x))
         &alpha,              # np.float64_t *alpha, # Scalar multiple
         &(Ai[0,0]),          # np.float64_t *a, # Matrix A: mxn
         &nb,                 # int *lda, # The size of the first dimension of A (in memory)
         &(b[0]),             # np.float64_t *x, # Vector x, min(len(x)) = n
         &inc,                # int *incx, # The increment between elements of x (usually 1)
         &beta,               # np.float64_t *beta, # Scalar multiple
         &(res[0]),           # np.float64_t *y, # Vector y, min(len(y)) = m
         &inc                 # int *incy # The increment between elements of y (usually 1)
        )

    zinterp[0] = 0.0
    sigmasq[0] = 0.0
    for k in range(n):
        zinterp[0] += res[k]*z[k]
    for k in range(nb):
        sigmasq[0] += res[k]*(-b[k])

    return 0

# copied from variogram_model.py

#cdef void _c_linear_variogram_model(double [::1] params, np.ndarray[np.float64_t]  dist, np.ndarray[np.float64_t] out):
#    out[:] =  params[0]*dist + params[1]

cdef void _c_linear_variogram_model(double [::1] params, double [::1]  dist, int n, double[::1] out) nogil:
    cdef int k
    for k in range(n):
        out[k] =  params[0]*dist[k] + params[1]


cdef _c_power_variogram_model(params, dist, out):
    out[:] =  float(params[0])*(dist**float(params[1])) + float(params[2])


cdef _c_gaussian_variogram_model(params, dist, out):
    out[:] = (float(params[0]) - float(params[2]))*(1 - np.exp(-dist**2/(float(params[1])*4.0/7.0)**2)) + \
            float(params[2])


cdef _c_exponential_variogram_model(params, dist, out):
    out[:] = (float(params[0]) - float(params[2]))*(1 - np.exp(-dist/(float(params[1])/3.0))) + \
            float(params[2])


cdef _c_spherical_variogram_model(params, dist, out):
    out[:] =  np.piecewise(dist, [dist <= float(params[1]), dist > float(params[1])],
                        [lambda x: (float(params[0]) - float(params[2])) *
                                   ((3*x)/(2*float(params[1])) - (x**3)/(2*float(params[1])**3)) + float(params[2]),
                         float(params[0])])
