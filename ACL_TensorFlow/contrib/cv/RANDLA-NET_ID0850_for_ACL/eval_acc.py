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
#
# Author: Qingyong Hu (huqingyong15@outlook.com) 15/11/2019


import numpy as np
from sklearn.metrics import confusion_matrix
# import open3d
# import colorsys, random


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU
    return IoU


#     ##################
#     # Visualization data function #
#     ##################
# class Plot:
#     def random_colors(N, bright=True, seed=0):
#         brightness = 1.0 if bright else 0.7
#         hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
#         colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#         random.seed(seed)
#         random.shuffle(colors)
#         return colors
#
#     def draw_pc(pc_xyzrgb):
#         pc = open3d.PointCloud()
#         pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
#         if pc_xyzrgb.shape[1] == 3:
#             open3d.draw_geometries([pc])
#             return 0
#         if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
#             pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
#         else:
#             pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6])
#         open3d.draw_geometries([pc])
#         return 0
#
#     def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
#         """
#         pc_xyz: 3D coordinates of point clouds
#         pc_sem_ins: semantic or instance labels
#         plot_colors: custom color list
#         """
#         if plot_colors is not None:
#             ins_colors = plot_colors
#         else:
#             ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)
#
#         ##############################
#         sem_ins_labels = np.unique(pc_sem_ins)
#         sem_ins_bbox = []
#         Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
#         for id, semins in enumerate(sem_ins_labels):
#             valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
#             if semins <= -1:
#                 tp = [0, 0, 0]
#             else:
#                 if plot_colors is not None:
#                     tp = ins_colors[semins]
#                 else:
#                     tp = ins_colors[id]
#
#             Y_colors[valid_ind] = tp
#
#             ### bbox
#             valid_xyz = pc_xyz[valid_ind]
#
#             xmin = np.min(valid_xyz[:, 0]);
#             xmax = np.max(valid_xyz[:, 0])
#             ymin = np.min(valid_xyz[:, 1]);
#             ymax = np.max(valid_xyz[:, 1])
#             zmin = np.min(valid_xyz[:, 2]);
#             zmax = np.max(valid_xyz[:, 2])
#             sem_ins_bbox.append(
#                 [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])
#
#         Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
#         Plot.draw_pc(Y_semins)
#         return Y_semins


labels_file = "/root/bin_out/labels.bin"
stacked_probs_file = "/root/bin_out/randlanet_output_0.bin"
pc_xyz_file = "/root/bin_out/xyz_0.bin"
num_classes = 13
batch_size = 3
point_num = 40960

label_to_names = {0: 'ceiling',
                  1: 'floor',
                  2: 'wall',
                  3: 'beam',
                  4: 'column',
                  5: 'window',
                  6: 'door',
                  7: 'table',
                  8: 'chair',
                  9: 'sofa',
                  10: 'bookcase',
                  11: 'board',
                  12: 'clutter'}
label_values = np.sort([k for k, v in label_to_names.items()])
confusion_list = []

stacked_labels = np.array(np.fromfile(labels_file, dtype=np.int32))
stacked_probs = np.array(np.fromfile(stacked_probs_file, dtype=np.float32)).reshape(len(stacked_labels), -1)

# Number of points per class in validation set
val_proportions = np.zeros(num_classes, dtype=np.float32)
i = 0
for label_val in label_values:
    val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in stacked_labels])
    i += 1

correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
acc = correct / float(np.prod(np.shape(stacked_labels)))
print(' acc:' + str(acc))

preds = label_values[np.argmax(stacked_probs, axis=1)].astype(np.int32)
labels = stacked_labels

# Confs
confusion_list += [confusion_matrix(labels, preds, label_values)]
# Regroup confusions
C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
# Rescale with the right number of point per class
C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

# Compute IoUs
IoUs = IoU_from_confusions(C)
m_IoU = np.mean(IoUs)
s = '{:5.2f} | '.format(100 * m_IoU)
for IoU in IoUs:
    s += '{:5.2f} '.format(100 * IoU)
print(s)

#     ##################
#     # Visualize raw data and result #
#     ##################
# pc_xyz = np.array(np.fromfile(pc_xyz_file, dtype=np.float32)).reshape(batch_size, point_num, -1)
# labels = labels.reshape(batch_size, -1)
# pres_labels = np.argmax(stacked_probs, axis=1).reshape(batch_size, -1)
# Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :]) # Raw data
# Plot.draw_pc_sem_ins(pc_xyz[0, :, :], pres_labels[0, :]) # Predicted result
