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
from packages.lifting import utils
import numpy as np

b_image = np.fromfile('./input/b_image.bin', np.float32).reshape(1, 368, 654, 3)

heatmap_person_large = np.fromfile('./output/20220318_172255/hmap_output_0.bin', np.float32).reshape(368, 654)
heatmap_person_large = np.squeeze(heatmap_person_large)

centers = utils.detect_objects_heatmap(heatmap_person_large)

b_pose_image, b_pose_cmap = utils.prepare_input_posenet(
    b_image[0], centers,
    [utils.config.INPUT_SIZE, 654],
    [utils.config.INPUT_SIZE, utils.config.INPUT_SIZE],
    batch_size=utils.config.BATCH_SIZE)

b_pose_image = b_pose_image.astype('float32')
b_pose_image.tofile('./input/b_pose_image.bin')
b_pose_cmap = b_pose_cmap.astype('float32')
b_pose_cmap.tofile('./input/b_pose_cmap.bin')
