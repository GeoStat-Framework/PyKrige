# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
ctypedef void (*variogram_model_t)(double [::1], long, double [::1], double [::1])

cdef variogram_model_t get_variogram_model(function_name)
