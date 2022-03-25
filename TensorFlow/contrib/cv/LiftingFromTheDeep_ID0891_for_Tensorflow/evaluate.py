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
from packages.lifting.utils import gaussian_heatmap, config
import cv2
import argparse
import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


# set up argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./dataset/MPII')     # MPII dataset root
parser.add_argument('--label_path', type=str,
                    default='./dataset/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')  #label path
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/model.ckpt')     # checkpoint path
parser.add_argument('--prob_model_path', type=str,
                    default='./data/prob_model/prob_model_params.mat')     # 3d model path

args = parser.parse_args()
input_width = 654
input_height = 368


def save_joints(): # read mpii dataset and label
    mat = loadmat(args.label_path)
    d = {}
    fd = []
    for i, (anno, train_flag) in enumerate(
            zip(mat['RELEASE']['annolist'][0, 0][0],
                mat['RELEASE']['img_train'][0, 0][0],
                )):
        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            # only one person
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            datas = []
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]
                    # build feed_dict
                    feed_dict = {}
                    feed_dict['width'] = int(abs(float(head_x2[0, 0]) - float(head_x1[0, 0])))
                    feed_dict['height'] = int(abs(float(head_y2[0, 0]) - float(head_y1[0, 0])))

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0] for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v for k, v in zip(j_id, vis)])
                    else:
                        vis = None
                    feed_dict['x'] = x
                    feed_dict['y'] = y
                    feed_dict['vis'] = vis
                    feed_dict['filename'] = img_fn
                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }
                        datas.append(data)

            for data in datas:
                head_center = [(data['head_rect'][0] + data['head_rect'][2]) / 2, (data['head_rect'][1] + data['head_rect'][3]) / 2]
                if d.get(data['filename']):
                    d.get(data['filename']).append(data)
                else:
                    d[data['filename']] = [data]
    filt = []
    for key, value in d.items():
        if len(value) != 1:
            filt.append(key)
    for key in filt:
        del d[key]
    return d


def generate_center_map(center_poses, img_shape): # input label position and generate a heat-map
    """
    Given the position of the person and the size of the input image it
    generates
    a heat-map where a gaissian distribution is fit in the position of the
    person in the image.
    """
    img_height = img_shape[1]
    img_width = img_shape[0]
    # Gaussian operator generate a heat-map
    center_map = [gaussian_heatmap(
        img_height, img_width, center_poses[1], center_poses[0],
        config.SIGMA_CENTER, config.SIGMA_CENTER)]

    out = np.zeros_like(center_map[0])
    # multiple map composition
    for map in center_map:
        out += map
    out[out > 1] = 1
    return out


def preprocess(k, input_width=654, input_height=368):   # read image pretreatment
    # read image
    image = cv2.imread(os.path.join(args.data_path, 'images', k))
    ratio = (input_width / image.shape[1], input_height / image.shape[0])
    image = cv2.resize(image, (input_width, input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    # obtain label
    labels = [d[k][0]['joint_pos']['7'][0] * 0.5 + d[k][0]['joint_pos']['6'][0] * 0.5, d[k][0]['joint_pos']['7'][1] * 0.5 + d[k][0]['joint_pos']['6'][1] * 0.5]
    labels[0] *= ratio[0]
    labels[1] *= ratio[1]
    # obtain headSize
    headSize = d[k][0]['head_rect']
    headSize = (headSize[2] - headSize[0]) * 0.5 + (headSize[3] - headSize[1]) * 0.5
    heatmap_gt = generate_center_map(labels, (input_width, input_height))# generate a heat-map
    return image, labels, heatmap_gt, headSize

def get_batch(idxs):    # read batch data
    name_lst = np.array(list(d.keys()))[idxs]
    images = []
    labels = []
    heatmap_gts = []
    headSizes = []
    for name in name_lst:
        image, label, heatmap_gt, headSize = preprocess(name)
        images.append(image)
        labels.append(label)
        heatmap_gts.append(heatmap_gt)
        headSizes.append(headSize)
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    heatmap_gts = np.stack(heatmap_gts, 0)
    headSizes = np.stack(headSizes, 0)
    return images, labels, heatmap_gts, headSizes


def calCKh(pred, label, headSize):
    dist = np.sqrt(np.sum((np.array(pred) - np.array(label)) ** 2)) / headSize
    CKh = 1 if dist < 0.5 else 0
    # print(Chk)
    return CKh

def main():
    # generate batch

    batch_idxs = np.random.permutation(len(d))
    batch_idxs = np.array_split(batch_idxs, len(d))

    # model definition
    pose_estimator = PoseEstimator((input_height, input_width, 3), args.checkpoint_path, args.prob_model_path)

    # model initialisation
    pose_estimator.initialise()

    # validation
    CKh_num = 0
    for i, idxs in enumerate(tqdm(batch_idxs)):
        # generate batch
        images, labels, heatmap_gts, headSizes = get_batch(idxs)
        pose_2d, heatmap_pred = pose_estimator.estimate(images[0])

        if len(pose_2d) < 1:
            continue
        pred = [pose_2d[0, 8, 1] * 0.25 + pose_2d[0, 11, 1] * 0.25 + pose_2d[0, 1, 1] * 0.5,
                pose_2d[0, 8, 0] * 0.25 + pose_2d[0, 11, 0] * 0.25 + pose_2d[0, 1, 0] * 0.5]

        CKh = calCKh(pred, labels[0], headSizes[0])
        CKh_num += CKh
    PCKh = CKh_num / len(batch_idxs)
    print('PCKh@0.5: ', PCKh)

    # close model
    pose_estimator.close()


d = save_joints()
if __name__ == '__main__':
    import sys
    sys.exit(main())

