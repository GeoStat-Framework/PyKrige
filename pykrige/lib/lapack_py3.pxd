cimport numpy as np

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

ctypedef int dgesv_t(
        # Solve A*x=b
        int *n, #  The number of linear equations,
        int *nrhs, #  The number of right hand sides, i.e.,  the  number of columns of the matrix B.  NRHS >= 0.
        np.float64_t *a, # Matrix A: mxn
        int *lda, # The size of the first dimension of A (in memory)
        int *ipiv, # Integer array
        np.float64_t *b, # Vector b
        int *ldb, # The size of the first dimension of A (in memory)
        int *info, # The size of the first dimension of A (in memory)
        )

cdef dgemv_t *dgemv

cdef dgesv_t *dgesv
