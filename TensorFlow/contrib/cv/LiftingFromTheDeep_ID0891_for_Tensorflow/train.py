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
from scipy.io import loadmat
from packages.lifting import PoseEstimator
from packages.lifting.utils import gaussian_heatmap, config, plot_pose, draw_limbs
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

# set up argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./dataset/MPII')     # dataset path
parser.add_argument('--batch_size', type=int, default=4)        # batchsize
parser.add_argument('--save_step', type=int, default=5)     # model saved interval
parser.add_argument('--epochs', type=int, default=30)       # train epoch
parser.add_argument('--output_path',type=str,           # where to save checkpoint
                    default='./checkpoint/model.ckpt')

parser.add_argument('--label_path', type=str,
                    default='./dataset/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')  #label path
parser.add_argument('--prob_model_path', type=str,
                    default='./data/prob_model/prob_model_params.mat')     # 3d model path
parser.add_argument('--init_session_path',type=str,
                    default='./data/init_session/init')
args = parser.parse_args()



input_width = 654
input_height = 368

#if not os.path.exists(OUT_SESSION_PATH):
#    os.mkdir(OUT_SESSION_PATH)

def save_joints(): # read mpii dataset image and label
    mat = loadmat(args.label_path)
    d = {}
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
    # obtain headsize
    headsize = d[k][0]['head_rect']
    headsize = (headsize[2] - headsize[0]) * 0.5 + (headsize[3] - headsize[1]) * 0.5
    heatmap_gt = generate_center_map(labels, (input_width, input_height))# generate a heat-map
    return image, labels, heatmap_gt, headsize


def get_batch(idxs):
    name_lst = np.array(list(d.keys()))[idxs]
    images = []
    labels = []
    heatmap_gts = []
    headsizes = []
    for name in name_lst:
        image, label, heatmap_gt, headsize = preprocess(name)
        images.append(image)
        labels.append(label)
        heatmap_gts.append(heatmap_gt)
        headsizes.append(headsize)
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    heatmap_gts = np.stack(heatmap_gts, 0)
    headsizes = np.stack(headsizes, 0)
    return images, labels, heatmap_gts, headsizes


def calCKh(pred, label, headsize):
    dist = np.sqrt(np.sum((np.array(pred) - np.array(label)) ** 2)) / headsize
    CKh = 1 if dist < 0.5 else 0
    # print(Chk)
    return CKh

def shuffle_batch():
    batch_size = args.batch_size
    # generate batch

    batch_idxs = np.random.permutation(len(d))
    np.random.shuffle(batch_idxs)

    # 10-fold cross-validation
    num_train_idxs = (len(d) * 9 // (batch_size * 10)) * batch_size
    
    train_batch_idxs = batch_idxs[:num_train_idxs]
    train_batch_idxs = np.array_split(train_batch_idxs, len(train_batch_idxs) // batch_size)
    test_batch_idxs = batch_idxs[num_train_idxs: ]
    test_batch_idxs = np.array_split(test_batch_idxs, len(test_batch_idxs) // 1)

    return train_batch_idxs, test_batch_idxs


def main():
    # define model
    pose_estimator = PoseEstimator((input_height, input_width, 3), args.init_session_path, args.prob_model_path)
    # initialization
    pose_estimator.initialise()

    train_batch_idxs, test_batch_idxs = shuffle_batch()

    # start training
    epochs = args.epochs
    print('Start training!')
    for epoch in range(epochs):
        train_losses = 0
        for i, idxs in enumerate(tqdm(train_batch_idxs)):
            images, labels, heatmap_gts, headsizes = get_batch(idxs)
            # input network training
            train_loss, heatmap_pred = pose_estimator.train(images, heatmap_gts)
            train_losses += train_loss
        print('Epoch {}: loss={}'.format(epoch, train_losses))

        if (epoch+1) % args.save_step == 0:
            pose_estimator.saver.save(pose_estimator.session, args.output_path) # save checkpoint
            print('Checkpoint saved successfully!')
            print('Start validation!')
            # validation
            CKh_num = 0
            for i, idxs in enumerate(test_batch_idxs):
                # generate batch
                images, labels, heatmap_gts, headsizes = get_batch(idxs)
                # input network training
                pose_2d, heatmap_pred = pose_estimator.estimate(images[0])

                if len(pose_2d) < 1:
                    continue
                pose_2d = [pose_2d[0, 8, 1] * 0.25 + pose_2d[0, 11, 1] * 0.25 + pose_2d[0, 1, 1] * 0.5,
                           pose_2d[0, 8, 0] * 0.25 + pose_2d[0, 11, 0] * 0.25 + pose_2d[0, 1, 0] * 0.5]
                CKh = calCKh(pose_2d, labels[0], headsizes[0])
                CKh_num += CKh
            PCKh = CKh_num / len(test_batch_idxs)
            print('Epoch {}: Validation PCKh@0.5:{} '.format(epoch, PCKh))

            train_batch_idxs, test_batch_idxs = shuffle_batch()

    # close model
    pose_estimator.close()
d = save_joints()

if __name__ == '__main__':
    import sys

    sys.exit(main())

