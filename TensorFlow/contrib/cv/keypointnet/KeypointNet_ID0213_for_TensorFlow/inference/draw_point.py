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
# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=str, default=None)
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

num_kp = 10


def draw_circle(rgb, u, v, col, r):
  """Draws a simple anti-aliasing circle in-place.

  Args:
    rgb: Input image to be modified.
    u: Horizontal coordinate.
    v: Vertical coordinate.
    col: Color.
    r: Radius.
  """

  ir = int(math.ceil(r))
  for i in range(-ir-1, ir+2):
    for j in range(-ir-1, ir+2):
      nu = int(round(u + i))
      nv = int(round(v + j))
      if nu < 0 or nu >= rgb.shape[1] or nv < 0 or nv >= rgb.shape[0]:
        continue

      du = abs(nu - u)
      dv = abs(nv - v)

      # need sqrt to keep scale
      t = math.sqrt(du * du + dv * dv) - math.sqrt(r * r)
      if t < 0:
        rgb[nv, nu, :] = col
      else:
        t = 1 - t
        if t > 0:
          # t = t ** 0.3
          rgb[nv, nu, :] = col * t + rgb[nv, nu, :] * (1-t)
  
  return rgb


def draw_ndc_points(rgb, xy, cols):
  """Draws keypoints onto an input image.

  Args:
    rgb: Input image to be modified.
    xy: [n x 2] matrix of 2D locations.
    cols: A list of colors for the keypoints.
  """

  vh, vw = rgb.shape[0], rgb.shape[1]

  for j in range(len(cols)):
    x, y = xy[j, :2]
    x = (min(max(x, -1), 1) * vw / 2 + vw / 2) - 0.5
    y = vh - 0.5 - (min(max(y, -1), 1) * vh / 2 + vh / 2)

    x = int(round(x))
    y = int(round(y))
    if x < 0 or y < 0 or x >= vw or y >= vh:
      continue

    rad = 1.5
    rad *= rgb.shape[0] / 128.0
    rgb = draw_circle(rgb, x, y, np.array([0.0, 0.0, 0.0, 1.0]), rad * 1.5)
    rgb = draw_circle(rgb, x, y, cols[j], rad)
  
  misc.imsave(args.output, rgb)
  print("Finish drawing keypoint into the input image.")


cols = plt.cm.get_cmap("rainbow")(
      np.linspace(0, 1.0, num_kp))[:, :4]

rgb = misc.imread(args.input).astype(np.float32) / 255
xy = np.fromfile(args.bin, dtype='float32').reshape(num_kp, 2)

draw_ndc_points(rgb, xy, cols)