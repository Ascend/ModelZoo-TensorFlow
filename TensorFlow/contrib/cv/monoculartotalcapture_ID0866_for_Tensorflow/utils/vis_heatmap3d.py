# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import utils.general


class vis_heatmap3d(object):
    def __init__(self, fig, ax, heatmap, keypoints=None, type_str=None):
        assert len(heatmap.shape) == 4
        self.fig = fig
        self.idx = 0
        self.threshold = 0.5
        self.heatmap = heatmap
        self.ax = ax
        self.keypoints = keypoints
        self.type_str = type_str

        axcolor = 'lightgoldenrodyellow'
        axx = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

        self.slider_threshold = Slider(axx, 'threshold', 0.0, 1.0, valinit=0.5)
        self.slider_threshold.on_changed(self.update)

    def draw(self):
        self.ax.clear()
        if self.keypoints is not None:
            utils.general.plot3d(self.ax, self.keypoints, self.type_str)
        active_map = self.heatmap[:, :, :, self.idx]
        Z, Y, X = np.where(active_map >= self.threshold)
        colors = [(1 - s) * np.array([0., 0., 1.], dtype=float) for s in active_map[Z, Y, X]]
        self.ax.scatter(X, Y, Z, color=colors)

    def update(self, val):
        self.threshold = self.slider_threshold.val
        self.draw()
        self.fig.canvas.draw_idle()

