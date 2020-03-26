#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from libc.math cimport exp

# copied from variogram_model.py



cdef variogram_model_t get_variogram_model(name):
    cdef variogram_model_t c_func

    if name == 'linear_variogram_model':
        c_func = &_c_linear_variogram_model
    elif name == 'power_variogram_model':
        c_func = &_c_power_variogram_model
    elif name == 'gaussian_variogram_model':
        c_func = &_c_gaussian_variogram_model
    elif name == 'exponential_variogram_model':
        c_func = &_c_exponential_variogram_model
    elif name == 'spherical_variogram_model':
        c_func = &_c_spherical_variogram_model
    else:
        raise NotImplementedError

    return c_func


cdef void _c_linear_variogram_model(double [::1] params, long n, double [::1]  dist, double[::1] out) nogil:
    cdef long k
    cdef double a, b
    a = params[0]
    b = params[1]
    for k in range(n):
        out[k] = a*dist[k] + b


cdef void _c_power_variogram_model(double [::1] params, long n, double [::1] dist, double[::1] out) nogil:
    cdef long k
    cdef double a, b, c
    a = params[0]
    b = params[1]
    c = params[2]
    for k in range(n):
        out[k] =  a*(dist[k]**b) + c


cdef void _c_gaussian_variogram_model(double [::1] params, long n, double [::1]  dist, double [::1] out) nogil:
    cdef long k
    cdef double a, b, c
    a = params[0]
    b = params[1]
    c = params[2]
    for k in range(n):
        out[k] = (a - c)*(1 - exp(-(dist[k]/(b*4.0/7.0))**2)) + c


cdef void _c_exponential_variogram_model(double [::1] params, long n, double[::1] dist, double[::1] out) nogil:
    cdef long k
    cdef double a, b, c
    a = params[0]
    b = params[1]
    c = params[2]
    for k in range(n):
        out[k] = (a - c)*(1 - exp(-dist[k]/(b/3.0))) + c


cdef void _c_spherical_variogram_model(double [::1] params, long n, double[::1] dist, double[::1] out) nogil:
    cdef long k
    cdef double a, b, c
    a = params[0]
    b = params[1]
    c = params[2]
    for k in range(n):
        if dist[k] < b:
            out[k] = (a - c)*((3*dist[k])/(2*b) - (dist[k]**3)/(2*b**3)) + c
        else:
            out[k] = a
