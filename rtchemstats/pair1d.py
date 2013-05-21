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

from functools import wraps

import numpy as np

from .cython.pair1d import BaseStaticIsotropicPairCorrelationComputer
from .util import cached_property



class StaticIsotropicPairCorrelation(object):
    '''Describes the static correlation of isotropic particle pairs.
          e.g. The numerical analogs of the standard g(r) and h(r) functions.

       The underlying data is a histogram of pair separation distances at
       fixed spacing dr (i.e. the i-th bin contains the total number of
       pair observed at distances of i*dr <= r < (i+1)*dr.

       Additionally, the separation distances can be shifted by an r_offset.
       This is useful in avoiding the avoid the leading zeros that correspond
       to unphysically close pair distances.
    '''

    def __init__(self, pair_distance_histogram, dr, r_offset=0.0):
        pair_distance_histogram = np.asarray(pair_distance_histogram)
        assert pair_distance_histogram.ndim == 1
        assert pair_distance_histogram.size > 0
        self.pair_distance_histogram = pair_distance_histogram
        self.dr = dr
        self.r_offset = r_offset

    @cached_property
    def r_lower(self):
        '''Radial distance corresponding to correlation distances at
           the lower bound of each bin.
        '''
        return self.r_offset + self.dr * np.arange(self.pair_distance_histogram.size)

    @cached_property
    def r_mid(self):
        '''Radial distance corresponding to correlation distances at
           center of each bin.
        '''
        return self.r_lower + 0.5 * self.dr

    @property
    def r(self):
        return self.r_mid

    @cached_property
    def g(self):
        '''Reducued density pair correlations
        '''
        N = self.pair_distance_histogram.sum()
        if not N:
            return None

        r_max = self.pair_distance_histogram.size * self.dr
        V = 4.0 / 3.0 * np.pi * ((self.r_offset + r_max)**3 - self.r_offset**3)
        rho = N / V
        v = 4.0 / 3.0 * np.pi * ((self.r_lower + self.dr)**3 - self.r_lower**3)
        rhos =  self.pair_distance_histogram / v
        return rhos / rho

    @cached_property
    def h(self):
        '''Shifted reducued density pair correlations
        '''
        if self.g is None:
            return None
        return self.g - 1.0


class StaticIsotropicPairCorrelationComputer(BaseStaticIsotropicPairCorrelationComputer):
    '''Calculate the PairCorrelationData from configurations (Config) of
       particles. The calculation is performed by accumulating a histogram
       of pair separation distances one configuration at a time. The
       intermediate state of calculation can be saved by pickeling the
       calculator object.
    '''

    def analyze_config(self, config):
        self.analyze_positions(config.positions, config.box_size)

    def get_accumulated(self):
        return StaticIsotropicPairCorrelation(self.bins.copy(), self.dr, self.r_min)


class StaticIsotropicPairCorrelationIntegrator(object):
    '''Calculate thermodynamic properties of an isotropic particle system
       by integrating over the sampled pair correlation
       (StaticIsotropicPairCorrelation object) for the system.
       Uses the potential and gradient of the force field associated
       with the system.
    '''

    def __init__(self, pair_correlation, forcefield, rho, beta):
        self.pair_correlation = pair_correlation
        self.forcefield = forcefield
        self.rho = rho
        self.beta = beta

    def integrate_g_product_over_space_ex(self, func, mask=Ellipsis):
        '''Numerically integrate
            \int g(r) * r**2 * func(r)

           The mask argument allows the specification of which
           elements of data arrays to include (i.e. allows the
           exclusion of zero elements)
        '''
        r = self.pair_correlation.r[mask]
        g = self.pair_correlation.g[mask]
        return np.trapz(r**2 * g * func(r), r)

    @cached_property
    def where_g_nonzero(self):
        return self.pair_correlation.g != 0.0

    @cached_property
    def where_g_nonzero_and_in_cutoff(self):
        return (self.pair_correlation.g != 0.0) & (self.pair_correlation.r <= self.forcefield.r_cutoff)

    def integrate_g_product_over_space(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_g_nonzero)

    def integrate_g_product_over_space_in_cutoff(self, func):
        return self.integrate_g_product_over_space_ex(func, self.where_g_nonzero_and_in_cutoff)

    def calculate_excess_internal_energy(self):
        return (2.0 * np.pi * self.rho *
                self.integrate_g_product_over_space_in_cutoff(self.forcefield.evaluate_potential_function))

    def calculate_virial(self, correct_long_range=True):
        v = 1.0 - 2.0 / 3.0 * np.pi * self.beta * self.rho * (
            self.integrate_g_product_over_space_in_cutoff(
            lambda r: r * -self.forcefield.evaluate_scalar_force_function(r)))
        if correct_long_range:
            v += self.forcefield.long_range_virial_correction(self.pair_correlation.r.max())
        return v





