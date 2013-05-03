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

from .cython.tcf import (BaseMeanSquareDisplacementComputer,
                             BaseVelocityAutocorrelationComputer)

class BaseTimeCorelationComputer(object):
    '''Base class for Time Correlation Function (TCF) computations
    '''

    def __init__(self, window_size, N_particles, *args, **kwds):
        assert 'analyze_rate' in kwds
        self.analyze_rate = kwds.pop('analyze_rate')
        super(BaseTimeCorelationComputer, self).__init__(window_size, N_particles, *args, **kwds)

    @classmethod
    def create(cls, window_size, N_particles, analyze_rate=1):
        # ensure analyze_rate is passed as a keyword
        return cls(window_size, N_particles, analyze_rate=analyze_rate)

    def compute_time(self):
        return self.analyze_rate * np.arange(self.window_size)


class MeanSquareDisplacementComputer(BaseTimeCorelationComputer, BaseMeanSquareDisplacementComputer):
    '''Compute the mean square displacment TCF; i.e. th self-positional TCF
    '''

    def analyze_config(self, config):
        self.analyze_positions(config.positions, config.box_size)

    def compute_msd(self):
        n_acc = self.calculate_n_accumulates()
        if not n_acc:
            return None

        return self.acc_msd_data / float(n_acc * self.N_particles)

    def __reduce__(self):
        return (create_msdc,
                (self.window_size, self.N_particles,
                 self.analyze_rate,
                 self.n_positions_seen,
                 self.displacement_window, self.last_positions,
                 self.acc_msd_data))

def create_msdc(window_size, N_particles, analyze_rate, n_positions_seen,
                displacement_window, last_positions, acc_msd_data):
    return MeanSquareDisplacementComputer(window_size, N_particles,
                                            analyze_rate=analyze_rate,
                                            n_positions_seen=n_positions_seen,
                                            displacement_window=displacement_window,
                                            last_positions=last_positions,
                                            acc_msd_data=acc_msd_data)


class VelocityAutocorrelationComputer(BaseTimeCorelationComputer, BaseVelocityAutocorrelationComputer):
    '''Compute the velocity autocorrelation TCF (VACF); i.e. the self-velocity TCF
    '''

    def analyze_config(self, config):
        self.analyze_velocities(config.velocities)

    def compute_vacf(self):
        n_acc = self.calculate_n_accumulates()
        if not n_acc:
            return None

        return self.acc_correlations / float(n_acc * self.N_particles)

    def __reduce__(self):
        return (create_vacfc,
                (self.window_size, self.N_particles,
                 self.analyze_rate,
                 self.n_velocities_seen,
                 self.velocities_windows, self.acc_correlations))

def create_vacfc(window_size, N_particles, analyze_rate, n_velocities_seen,
                 velocities_windows, acc_correlations):
    return VelocityAutocorrelationComputer(window_size, N_particles,
                                             analyze_rate=analyze_rate,
                                             n_velocities_seen=n_velocities_seen,
                                             velocities_windows=velocities_windows,
                                             acc_correlations=acc_correlations)
