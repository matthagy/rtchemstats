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

from .cython.bond import calculate_bonds_set


class HistBuilder(object):

    def __init__(self, low, high, N_bins):
        assert high > low
        assert N_bins >= 1
        self.low = low
        self.high = high
        self.N_bins = N_bins
        self.count = np.zeros(N_bins, dtype=int)
        self.dx = (self.high - self.low) / self.N_bins
        assert self.dx > 0

    def record(self, x):
        x = np.array(x, copy=True)
        x -= self.low
        x /= self.dx
        index = np.floor(x).astype(int)
        N = self.N_bins
        count = self.count
        for i in index:
            if (0 <= i < N):
                count[i] += 1

    def get_count(self):
        return self.count.copy()

    def get_low_bounds(self):
        return self.low + self.dx * np.arange(self.N_bins)

    def get_mid_bounds(self):
        return self.low + self.dx * (0.5 + np.arange(self.N_bins))


class LogHistBuilder(HistBuilder):

    def __init__(self, low, high, N_bins):
        super(LogHistBuilder, self).__init__(np.log(low), np.log(high), N_bins)

    def record(self, x):
        super(LogHistBuilder, self).record(np.log(x))

    def get_low_bounds(self):
        return np.exp(super(LogHistBuilder, self).get_low_bounds())

    def get_mid_bounds(self):
        return np.exp(super(LogHistBuilder, self).get_mid_bounds())


class BondDurationAnalyzer(object):

    def __init__(self, r_bond, dt, hist_builder):
        self.r_bond = r_bond
        self.dt = dt
        self.hist_builder = hist_builder
        self.N_analyze = 0
        self.start_steps = {}
        self.seen_first = False
        self.last_step = None
        self.analyze_rate = None

    def analyze(self, time_step, positions, box_size):
        current = calculate_bonds_set(positions, box_size, self.r_bond)
        start_steps = self.start_steps
        for bond in current:
            if bond not in start_steps:
                start_steps[bond] = time_step

        durations = []
        for bond in list(start_steps):
            if bond not in current:
                duration = time_step - start_steps[bond]
                assert duration > 0
                del start_steps[bond]
                durations.append(duration)
        self.hist_builder.record(durations)

        if not self.seen_first:
            self.seen_first = True
        else:
            self.N_analyze += 1
            delta = time_step - self.last_step
            if self.N_analyze == 1:
                self.analyze_rate = delta
            else:
                assert self.analyze_rate == delta
        self.last_step= time_step

    def get_frequencies(self):
        if self.analyze_rate is None:
            return None
        if self.N_analyze <= 0:
            return None

        total_time = self.analyze_rate * self.N_analyze * self.dt
        count = self.hist_builder.get_count()
        frequency = count / total_time
        return frequency

    def get_low_bounds(self):
        return self.dt * self.hist_builder.get_low_bounds()

    def get_mid_bounds(self):
        return self.dt * self.hist_builder.get_mid_bounds()


