#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport sqrt
cimport cython

from libc.math cimport floor, ceil
from libc.stdlib cimport free, malloc, realloc
from cpython cimport PyObject, Py_INCREF

from util cimport PyArray_DATA, c_periodic_distance_sqr

np.import_array()


cdef class BondArrayWrapper:

    cdef int* data_ptr
    cdef int length

    cdef set_data(self, int length, int* data_ptr):
        self.data_ptr = data_ptr
        self.length = length

    def __array__(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.length
        shape[1] = <np.npy_intp> 2
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(2, shape,
                                               np.NPY_INT,
                                               <void *>self.data_ptr)
        return ndarray

    def __dealloc__(self):
        free(<void*>self.data_ptr)

cdef wrap_bond_array(int length, int *data):
    cdef BondArrayWrapper array_wrapper = BondArrayWrapper()
    array_wrapper.set_data(length, data)
    cdef np.ndarray ndarray = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*>array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)
    return ndarray

def calculate_bonds(np.ndarray[double, ndim=2, mode='c'] positions not None,
                    double box_size,
                    double r_bond):
    assert positions.shape[1] == 3

    cdef double *positions_p = <double *>PyArray_DATA(positions)
    cdef int *bonds
    cdef int N_bonds, N_bonds_alloc
    cdef double r_bond2 = r_bond*r_bond

    cdef size_t N = positions.shape[0]
    cdef unsigned int i,j

    N_bonds = 0
    N_bonds_alloc = 1024
    bonds = <int *>malloc(2 * N_bonds_alloc * sizeof(int))

    for i in range(0, N, 1):
        for j in range(i+1, N, 1):
            r2 = c_periodic_distance_sqr(positions_p + 3*i, positions_p + 3*j, box_size)
            if r2 < r_bond2:
                if N_bonds == N_bonds_alloc:
                    N_bonds_alloc += 1024
                    bonds = <int *>realloc(bonds, 2*N_bonds_alloc*sizeof(int))
                bonds[2*N_bonds] = i
                bonds[2*N_bonds+1] = j
                N_bonds += 1

    return wrap_bond_array(N_bonds, bonds)

def calculate_bonds_set(np.ndarray[double, ndim=2, mode='c'] positions not None,
                    double box_size,
                    double r_bond):
    assert positions.shape[1] == 3

    cdef double *positions_p = <double *>PyArray_DATA(positions)
    cdef int *bonds
    cdef int N_bonds, N_bonds_alloc
    cdef double r_bond2 = r_bond*r_bond

    cdef size_t N = positions.shape[0]
    cdef unsigned int i,j

    cdef set acc = set()

    for i in range(0, N, 1):
        for j in range(i+1, N, 1):
            r2 = c_periodic_distance_sqr(positions_p + 3*i, positions_p + 3*j, box_size)
            if r2 < r_bond2:
                acc.add((i,j))
    return acc



