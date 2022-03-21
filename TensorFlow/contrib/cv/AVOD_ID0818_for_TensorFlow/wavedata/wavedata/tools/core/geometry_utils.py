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


def calculate_plane_point(plane, point):
    """Calculates the point on the 3D plane for a point with one value missing

    :param plane: Coefficients of the plane equation (a, b, c, d)
    :param point: 3D point with one value missing (e.g. [None, 1.0, 1.0])
    :return: Valid point on the plane
    """

    a, b, c, d = plane
    x, y, z = point

    if x is None:
        x = -(b*y + c*z + d) / a
    elif y is None:
        y = -(a*x + c*z + d) / b
    elif z is None:
        z = -(a*x + b*y + d) / c

    return [x, y, z]


def dist_to_plane(plane, points):
    """Calculates the signed distance from a 3D plane to each point in a list
    of points

    :param plane: Coefficients of the plane equation (a, b, c, d)
    :param points: List of points
    :return: Signed distance of each point to the plane
    """
    a, b, c, d = plane

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return (a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

