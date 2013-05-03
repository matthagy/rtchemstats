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

cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.math cimport floor, ceil, sqrt


from util cimport c_periodic_distance, PyArray_DATA


cdef class BaseStaticIsotropicPairCorrelationComputer:
    '''Base function for computing an isotropic static pair correlation function
    '''

    cdef public:
        double dr, r_min, r_max
        size_t N_bins
        np.ndarray bins

    def __cinit__(self, dr, r_max, r_min=0.0, bins=None):
        cdef int N_bins = <int>ceil((r_max - r_min) / dr)
        if N_bins <= 0:
            raise ValueError("bad parameters")

        self.dr = dr
        self.r_min = r_min
        self.r_max = r_max
        self.N_bins = <size_t>N_bins
        if bins is None:
            bins = np.zeros(self.N_bins, dtype=np.uint)
        if bins.shape != (self.N_bins,) or bins.dtype != np.uint:
            raise TypeError("bad bins argument")
        self.bins = bins

    def __reduce__(self):
        return self.__class__, (self.dr, self.r_max, self.r_min, self.bins.copy())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def accumulate_positions(self, np.ndarray[double, ndim=2, mode='c'] positions not None,
                             double box_size):
        cdef double *positions_p = <double *>PyArray_DATA(positions)
        cdef np.ndarray[unsigned long, ndim=1, mode='c'] bins = self.bins
        cdef size_t N3 = 3*positions.shape[0]
        cdef unsigned int i,j
        cdef int index
        cdef double r

        for i in range(0, N3, 3):
            for j in range(i+3, N3, 3):
                r = c_periodic_distance(positions_p + i, positions_p + j, box_size)
                index = <int>floor((r - self.r_min) / self.dr)
                if index >= 0 and index < self.N_bins:
                    bins[index] += 1
