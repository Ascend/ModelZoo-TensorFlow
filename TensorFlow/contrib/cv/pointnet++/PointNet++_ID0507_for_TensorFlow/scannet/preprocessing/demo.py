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
import sys
import os

BASE_DIR = os.path.dirname(__file__)

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

import numpy as np
import pc_util

data = np.load('scannet_scenes/scene0001_01.npy')
scene_points = data[:,0:3]
colors = data[:,3:6]
instance_labels = data[:,6]
semantic_labels = data[:,7]


output_folder = 'demo_output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Write scene as OBJ file for visualization
pc_util.write_ply_rgb(scene_points, colors, os.path.join(output_folder, 'scene.obj'))
pc_util.write_ply_color(scene_points, instance_labels, os.path.join(output_folder, 'scene_instance.obj'))
pc_util.write_ply_color(scene_points, semantic_labels, os.path.join(output_folder, 'scene_semantic.obj'))

