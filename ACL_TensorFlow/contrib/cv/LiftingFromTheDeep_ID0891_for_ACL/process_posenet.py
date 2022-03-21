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
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./input/test_image.jpg')
#image = cv2.resize(image, (654, 368))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

heatmap_person_large = np.fromfile('./output/20220318_172255/hmap_output_0.bin',np.float32).reshape(368,654)
pred_likelihood = np.fromfile('./output/20220318_172345/posenet_output_0.bin',np.float32).reshape(4, 14)
pred_2d_pose = np.fromfile('./output/20220318_172345/posenet_output_1.bin',np.int64).reshape(4, 2, 14)
prob_model_path = './packages/prob_model_params.mat'

centers = utils.detect_objects_heatmap(heatmap_person_large)
estimated_2d_pose, visibility = utils.detect_parts_from_likelihoods(pred_2d_pose, centers, pred_likelihood)


poseLifting = utils.Prob3dPose(prob_model_path)
transformed_pose2d, weights = poseLifting.transform_joints(
    estimated_2d_pose.copy(), visibility)

orig_img_size = np.array(image.shape)
scale = utils.config.INPUT_SIZE / (orig_img_size[0] * 1.0)

pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
pose_2d = np.round(estimated_2d_pose / scale).astype(np.int32)

# 2d result
plt.figure()
utils.draw_limbs(image, pose_2d, visibility)
plt.axis('off')
plt.imshow(image)
plt.savefig('./result/result2d.jpg')

# 3d result
for i, single_3D in enumerate(pose_3d):
    utils.plot_pose(single_3D)
    plt.savefig('./result/result3d_{}.jpg'.format(i))


