# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class MovingAvg:
    """Class to record the buffering running statistics.

    Args:
        smooth: float, a scalar in [0, 1] to smooth the statistics.

    Attributes:
        sum: summation of historical data.
        avg: average of historical data.
        smooth_avg: smoothed average of historical data.
        count: total number of historical data record.
        cur_val: current data.

    Raises:
        ValueError, when smooth is not between [0, 1].
    """
    def __init__(self, smooth=0.9):
        if not (0. <= smooth <= 1.):
            raise ValueError(f'Smooth value should be between [0, 1], '
                             f'but is given {smooth}.')
        self.smooth = smooth
        self.clear()

    def update(self, val):
        """Update statistics.
        """
        self.cur_val = val
        self.count += 1
        self.sum += val
        self.avg = self.sum / self.count
        if self.count == 1:
            self.smooth_avg = val
        else:
            self.smooth_avg = self.smooth * self.smooth_avg + (1. - self.smooth) * val

    def clear(self):
        """Clear all historical data.
        """
        self.sum = 0.
        self.avg = 0.
        self.smooth_avg = 0.
        self.count = 0
        self.cur_val = 0
