# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np

# copied from variogram_model.py



cdef variogram_model_t get_variogram_model(function_name):
    cdef variogram_model_t c_variogram_function

    if function_name == 'linear_variogram_model':
        c_variogram_function = &_c_linear_variogram_model
    else:
        raise NotImplementedError
    return c_variogram_function


cdef void _c_linear_variogram_model(double [::1] params, long n, double [::1]  dist, double[::1] out) nogil:
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
