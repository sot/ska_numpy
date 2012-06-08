import numpy as np
import cython
cimport numpy as np

DTYPE = np.int

ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE_double_t

@cython.wraparound(False)
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
                if vi <= a[ia]:
                    idx[iv] = ia
                    break
                else:
                    ia += 1
            else:
                idx[iv] = na
                break

    return idx
