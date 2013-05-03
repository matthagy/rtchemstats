#  Copyright (C) 2013 Matt Hagy <hagy@gatech.edu>
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
cimport libc.string as cstring

from util cimport (c_periodic_direction, wrapping_modulo,
                   PyArray_DATA)


cdef void copy_N3_array_data(np.ndarray dst, np.ndarray src, unsigned int N) nogil:
    cstring.memcpy(<void *>PyArray_DATA(dst),
                   <void *>PyArray_DATA(src),
                   N * 3 * sizeof(double))


cdef class BaseMeanSquareDisplacementCalculator:

    cdef readonly:
        unsigned int window_size, N_particles, n_positions_seen
        np.ndarray displacement_window, last_positions, sum_displacements, acc_msd_data

    def __cinit__(self, unsigned int window_size, unsigned int N_particles,
                  n_positions_seen=None,
                  np.ndarray[double, ndim=3] displacement_window = None,
                  np.ndarray[double, ndim=2] last_positions = None,
                  np.ndarray[double, ndim=1] acc_msd_data = None,
                  *args, **kwds):
        assert window_size > 0
        assert N_particles > 0

        if n_positions_seen is None:
            n_positions_seen = 0

        assert n_positions_seen >= 0

        self.window_size = window_size
        self.N_particles = N_particles
        self.n_positions_seen = n_positions_seen

        self.sum_displacements = np.empty((self.N_particles, 3), dtype=float)

        displacement_window_shape = (window_size, self.N_particles, 3)
        if displacement_window is not None:
            assert (<object>displacement_window).shape == displacement_window_shape
        else:
            displacement_window = np.empty(displacement_window_shape, dtype=float)
        self.displacement_window = displacement_window

        if last_positions is not None:
            assert (<object>last_positions).shape == (N_particles, 3)
        else:
            last_positions = np.empty((N_particles, 3), dtype=float)
        self.last_positions = last_positions

        if acc_msd_data is not None:
            assert (<object>acc_msd_data).shape == (window_size,)
        else:
            acc_msd_data = np.zeros(window_size, dtype=float)
        self.acc_msd_data = acc_msd_data

    def __init__(self, window_size, N_particles, *args, **kwds):
        assert np.allclose(window_size, self.window_size)
        assert np.allclose(N_particles, self.N_particles)

    def analyze_positions(self, np.ndarray[double, ndim=2] positions not None, double box_size):
        assert positions.shape[0] == self.N_particles
        assert positions.shape[1] == 3

        if self.n_positions_seen:
            self.insert_displacments(positions, box_size)

        copy_N3_array_data(self.last_positions, positions, self.N_particles)
        self.n_positions_seen += 1

        if self.n_positions_seen > self.window_size:
            self.accumulate_mean_square_displacments()

    def calculate_n_accumulates(self):
        return max(<int>self.n_positions_seen - <int>self.window_size, 0)

    cdef inline double *get_displacement_window(self, unsigned int window_index) nogil:
        # normalize index
        window_index = wrapping_modulo(self.n_positions_seen - 1 + window_index, self.window_size)
        return ((<double *>PyArray_DATA(self.displacement_window)) +
                (window_index * self.N_particles * 3))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void insert_displacments(self, np.ndarray positions, double box_size) nogil:
        '''Calculate displacement vectors from self.last_positions to positions.
           Store results in index 0 of the displacement window.
           Should be called before n_positions_seen is incremented so that
           we overwrite the oldest displacement window.
        '''
        cdef double *positions_p = <double *>PyArray_DATA(positions)
        cdef double *last_positions_p = <double *>PyArray_DATA(self.last_positions)
        cdef double *displacment_p = self.get_displacement_window(0)

        cdef unsigned int i
        for i in range(0, 3*self.N_particles, 3):
            c_periodic_direction(displacment_p + i, positions_p + i, last_positions_p + i, box_size)

    cdef void accumulate_mean_square_displacments(self) nogil:
        '''Move through displacement windows while accumulating
           the deltas to determine the displacement of each particle
           at each window in time.
        '''
        cdef double *sum_displacements_p = <double *>PyArray_DATA(self.sum_displacements)
        cdef double *acc_msd_data_p = <double *>PyArray_DATA(self.acc_msd_data)
        cdef double *displacement_slice
        cdef double delta, acc_sqr_delta
        cdef unsigned int N3 = 3*self.N_particles
        cdef unsigned int window_i, j

        cstring.memset(<void *>sum_displacements_p, 0, sizeof(double)*N3)

        for window_i in range(self.window_size):

            # Efficiently perform:
            #    self.sum_displacements += self.displacement_window[window_i]
            # while also accumulating square displacements

            acc_sqr_delta = 0
            displacement_slice = self.get_displacement_window(window_i)

            for j in range(N3):
                sum_displacements_p[j] += displacement_slice[j]
                delta = sum_displacements_p[j]
                acc_sqr_delta += delta*delta

            acc_msd_data_p[window_i] += acc_sqr_delta


cdef class BaseVelocityAutocorrelationCalculator:

    cdef readonly:
        unsigned int window_size, N_particles, n_velocities_seen
        np.ndarray velocities_windows, acc_correlations

    def __cinit__(self, unsigned int window_size, unsigned int N_particles,
                  n_velocities_seen=None,
                  np.ndarray[double, ndim=3] velocities_windows = None,
                  np.ndarray[double, ndim=1] acc_correlations = None,
                  *args, **kwds):
        assert window_size > 0
        assert N_particles > 0

        if n_velocities_seen is None:
            n_velocities_seen = 0

        assert n_velocities_seen >= 0

        self.window_size = window_size
        self.N_particles = N_particles
        self.n_velocities_seen = n_velocities_seen

        velocities_windows_shape = (self.window_size, self.N_particles, 3)
        if velocities_windows is not None:
            assert (<object>velocities_windows).shape == velocities_windows_shape
        else:
            velocities_windows = np.empty(velocities_windows_shape, dtype=float)
        self.velocities_windows = velocities_windows

        acc_correlations_shape = (self.window_size,)
        if acc_correlations is not None:
            assert (<object>acc_correlations).shape == acc_correlations_shape
        else:
            acc_correlations = np.zeros(acc_correlations_shape, dtype=float)
        self.acc_correlations = acc_correlations

    def __init__(self, window_size, N_particles, *args, **kwds):
        assert np.allclose(window_size, self.window_size)
        assert np.allclose(N_particles, self.N_particles)

    cdef inline double *get_velocities_window(self, unsigned int window_index) nogil:
        # normalize index
        window_index = wrapping_modulo(self.n_velocities_seen + window_index, self.window_size)
        return ((<double *>PyArray_DATA(self.velocities_windows)) +
                (window_index * self.N_particles * 3))

    def analyze_velocities(self, np.ndarray[double, ndim=2] velocities not None):
        assert velocities.shape[0] == self.N_particles
        assert velocities.shape[1] == 3

        # overwrite oldest velocities window with newest data
        cstring.memcpy(<void *>self.get_velocities_window(0),
                       <void *>PyArray_DATA(velocities),
                       self.N_particles * 3 * sizeof(double))

        self.n_velocities_seen += 1
        if self.n_velocities_seen >= self.window_size:
            self.accumulate_velocity_correlations()

    def calculate_n_accumulates(self):
        return max(<int>self.n_velocities_seen - <int>self.window_size + 1, 0)

    cdef void accumulate_velocity_correlations(self) nogil:
        cdef double *acc_correlations_p = <double *>PyArray_DATA(self.acc_correlations)
        cdef double *velocities_0, *velocities_i
        cdef double acc
        cdef unsigned int window_i, j
        cdef unsigned int N3 = 3*self.N_particles

        velocities_0 = self.get_velocities_window(0)

        for window_i in range(self.window_size):

            velocities_i = self.get_velocities_window(window_i)
            acc = 0
            for j in range(N3):
                acc += velocities_0[j] * velocities_i[j]
            acc_correlations_p[window_i] += acc






