cimport numpy as np
from libcpp.vector cimport vector

ctypedef fused FLOAT:
    np.float32_t
    np.float_t


cdef cppclass Pool:
    FLOAT v

    Pool() except +


cdef class VectorPool:
    cdef vector[Pool] P

    def push(self, x):
        cdef Pool P = Pool()