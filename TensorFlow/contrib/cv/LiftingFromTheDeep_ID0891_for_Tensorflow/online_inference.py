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
from npu_bridge.npu_init import *

from packages.lifting import PoseEstimator
from packages.lifting.utils import plot_pose, draw_limbs

import cv2
import matplotlib.pyplot as plt
import argparse
import os


# set up the argparse
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/model.ckpt')     # checkpoint path
parser.add_argument('--prob_model_path', type=str,
                    default='./data/prob_model/prob_model_params.mat')     # 3d model path
parser.add_argument('--test_image', type=str,
                    default='./dataset/MPII/images/099363014.jpg')
parser.add_argument('--result_path', type=str,
                    default='./result')

args = parser.parse_args()


def main():
    # read image
    image = cv2.imread(args.test_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    image_size = image.shape

    # initialize model
    pose_estimator = PoseEstimator(image_size, args.checkpoint_path, args.prob_model_path)

    # load model
    pose_estimator.initialise()

    # estimation
    pose_2d, visibility, pose_3d = pose_estimator.estimate(image, lifting=True)

    # Show 2D and 3D poses
    display_results(image, pose_2d, visibility, pose_3d)
    # close model
    pose_estimator.close()


def display_results(in_image, data_2d, joint_visibility, data_3d):  # 2d3d resalt visualization
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)

    plt.axis('off')
    # save 2d image
    plt.savefig(os.path.join(args.result_path,'result2d.jpg'))

    # Show 3D poses
    for i, single_3D in enumerate(data_3d):
        plot_pose(single_3D)
        plt.savefig(os.path.join(args.result_path, 'result3d_{}.jpg'.format(i))) # save images

if __name__ == '__main__':
    import sys
    sys.exit(main())
