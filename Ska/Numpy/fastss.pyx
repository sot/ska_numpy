import numpy as np
import cython
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE_double_t

@cython.boundscheck(False) 
def _search_both_sorted(np.ndarray[dtype=DTYPE_double_t, ndim=1] a not None,
                       np.ndarray[dtype=DTYPE_double_t, ndim=1] v not None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would
    be preserved.

    Parameters
    ----------
    a : 1-D array_like
        Input array, sorted in ascending order.
    v : array_like
        Values to insert into `a`.
    """
    cdef int nv = v.shape[0]
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] idx = np.empty(nv, dtype=DTYPE)
    cdef int na = a.shape[0]
    cdef unsigned int ia = 0
    cdef unsigned int iv
    cdef double vi

    for iv in range(nv):
        vi = v[iv]
        while True:
            if ia < na:
                if vi > a[ia]:
                    ia += 1
                else:
                    idx[iv] = ia
                    break
            else:
                if vi > a[na - 1]:
                    idx[iv] = na
                else:
                    idx[iv] = na - 1
                break

    return idx
