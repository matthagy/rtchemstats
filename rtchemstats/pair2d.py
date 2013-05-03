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

from cython.pair2d import HistAxis, Hist2DData, analyze_configuration



class Static2DPairCorrelation(object):

    def __init__(self, hist_cartessian, hist_angle):
        self.hist_cartessian = hist_cartessian
        self.hist_angle = hist_angle


class Static2DPairCorrelationComputer(object):

    def __init__(self, hist_cartessian, hist_angle):
        self.hist_cartessian = hist_cartessian
        self.hist_angle = hist_angle

    def analyze_positions_directors(self, positions, directors, box_size):
        analyze_configuration(positions, directors, box_size,
                              self.hist_cartessian,
                              self.hist_angle)

    def get_accumulated(self):
        return Static2DPairCorrelation(self.hist_cartessian.copy(),
                                       self.hist_angle.copy())
