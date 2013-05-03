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
from libc.math cimport floor, ceil, sqrt, acos, fabs
from util cimport c_vector_dot, c_periodic_direction

np.import_array()

cdef class HistAxis:

    cdef public:
        size_t N_bins
        double min, max, bin_width

    def __cinit__(self, size_t N_bins, double min, double max):
        assert N_bins >= 1
        assert min < max
        self.N_bins = N_bins
        self.min = min
        self.max = max
        self.bin_width = (max - min) / self.N_bins

    def __richcmp__(self, other, int op):
        if op != 2:
            return NotImplemented
        if not isinstance(other, HistAxis):
            return NotImplemented
        return (self.N_bins == other.N_bins and
                self.min == other.min and
                self.max == other.max)

    def __reduce__(self):
        return (self.__class__, (self.N_bins, self.min, self.max))

    @cython.cdivision(True)
    cdef size_t index(self, double x):
        return <size_t>floor((x - self.min) / self.bin_width)


cdef class Hist2DData:

    cdef public:
        HistAxis a1
        HistAxis a2
        np.ndarray count
        np.ndarray acc_value

    def __cinit__(self, HistAxis a1 not None, HistAxis a2=None, count=None, acc_value=None):
        if a2 is None:
            a2 = a1
        self.a1 = a1
        self.a2 = a2

        shape = self.a1.N_bins, self.a2.N_bins

        if count is not None:
            assert count.shape == shape
        else:
            count = np.zeros(shape, dtype=np.uint)

        if acc_value is not None:
            assert acc_value.shape == shape
        else:
            acc_value = np.zeros(shape, dtype=np.double)

        self.count = count
        self.acc_value = acc_value

    def __reduce__(self):
        return (self.__class__, (self.a1, self.a2, self.count, self.acc_value))

    def combine(self, other):
        if not isinstance(other, Hist2DData):
            raise TypeError
        assert self.a1 == other.a1
        assert self.a2 == other.a2
        return self.__class__(self.a1, self.a2,
                              self.count + other.count,
                              self.acc_value + other.acc_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline add_value(self, double x1, double x2, double value):
        cdef size_t i1 = self.a1.index(x1)
        cdef size_t i2 = self.a2.index(x2)
        self.count[i1, i2] += 1
        self.acc_value[i1, i2] += value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def analyze_configuration(np.ndarray[double, ndim=2, mode='c'] positions not None,
                          np.ndarray[double, ndim=2, mode='c'] orientations not None,
                          double box_size,
                          Hist2DData hist_cartessian,
                          Hist2DData hist_angle):
    cdef double *positions_p = <double *>np.PyArray_DATA(positions)
    cdef double *orientations_p = <double *>np.PyArray_DATA(orientations)
    cdef size_t N3 = 3*positions.shape[0]

    assert hist_angle.a1.min == 0
    assert np.allclose(hist_angle.a2.min, -np.pi/2)
    assert np.allclose(hist_angle.a2.max, np.pi/2)

    cdef unsigned int i,j
    cdef int index
    cdef double r_max = hist_angle.a1.max

    cdef double r[3]
    cdef double r2
    cdef double alpha
    cdef double ra, re, theta, r_length
    cdef double pi_2 = np.pi / 2

    for i in range(0, N3, 3):
        for j in range(0, N3, 3):

            if i==j:
                continue

            c_periodic_direction(r, positions_p + i, positions_p + j, box_size)
            r2 = c_vector_dot(r, r)

            ra = c_vector_dot(orientations_p + i, r)
            re = sqrt(r2 - ra*ra)

            if fabs(ra) > r_max or re > r_max:
                continue

            alpha = c_vector_dot(orientations_p + i, orientations_p + j)

            hist_cartessian.add_value(re, ra, alpha)
            r_length = sqrt(r2)

            if r_length <= r_max:
                theta = pi_2 - acos(ra / r_length)
                hist_angle.add_value(r_length, theta, alpha)




