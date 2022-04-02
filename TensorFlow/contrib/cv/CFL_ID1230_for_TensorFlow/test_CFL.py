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

import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import misc
from matplotlib import pyplot as plt
import imageio
from PIL import Image
import glob
import time
import math
import os.path
import Models
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from config import *

def preprocess(img):
    mean_color = [103.939, 116.779, 123.68]
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=img)
    bgr = tf.concat(values=[b - mean_color[0], g - mean_color[1], r - mean_color[2]], axis=3)
    return bgr


def evaluate(map):
    if map == 'edges':
        prediction_path_list = glob.glob(os.path.join(args.results, 'EM_test') + '/*.jpg')
        gt_path_list = glob.glob(os.path.join(args.dataset, 'EM_gt') + '/*.jpg')
    if map == 'corners':
        prediction_path_list = glob.glob(os.path.join(args.results, 'CM_test') + '/*.jpg')
        gt_path_list = glob.glob(os.path.join(args.dataset, 'CM_gt') + '/*.jpg')
    prediction_path_list.sort()
    gt_path_list.sort()

    P, R, Acc, f1, IoU = [], [], [], [], []
    prediction = Image.open(prediction_path_list[0])
    for im in range(len(prediction_path_list)):
        # predicted image
        prediction = Image.open(prediction_path_list[im])
        pred_W, pred_H = prediction.size
        prediction = np.array(prediction) / 255.
        # gt image
        gt = Image.open(gt_path_list[im])
        gt = gt.resize([pred_W, pred_H])
        gt = np.array(gt) / 255.
        gt = (gt >= 0.01).astype(int)

        th = 0.1
        tp = np.sum(np.logical_and(gt == 1, prediction > th))
        tn = np.sum(np.logical_and(gt == 0, prediction <= th))
        fp = np.sum(np.logical_and(gt == 0, prediction > th))
        fn = np.sum(np.logical_and(gt == 1, prediction <= th))

        # How accurate the positive predictions are
        P.append(tp / (tp + fp))
        # Coverage of actual positive sample
        R.append(tp / (tp + fn))
        # Overall performance of model
        Acc.append((tp + tn) / (tp + tn + fp + fn))
        # Hybrid metric useful for unbalanced classes
        f1.append(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        # Intersection over Union
        IoU.append(tp / (tp + fp + fn))

    return np.mean(P), np.mean(R), np.mean(Acc), np.mean(f1), np.mean(IoU)


def predict(image_path_list):
    rgb_ph1 = tf.compat.v1.placeholder(tf.float32, shape=(None, args.im_height, args.im_width, args.im_ch))
    rgb_ph = preprocess(rgb_ph1)

    net = Models.LayoutEstimator_StdConvs({'rgb_input': rgb_ph}, is_training=False)

    saver = tf.train.Saver()
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    with tf.Session(config=config) as sess:

        print('Loading the model')

        saver.restore(sess, args.weights)

        print('model loaded')

        # Obtain network predictions
        for image_path in image_path_list:

            name = str(image_path)
            filename = os.path.basename(name)

            img = Image.open(image_path)
            img = img.resize([args.im_width, args.im_height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)

            fd = net.fd_test
            fd[rgb_ph1] = img

            prediction = net.get_layer_output("output_likelihood")
            pred_edges, pred_corners = tf.split(prediction, [1, 1], 3)

            tt = time.time();
            emap, cmap = sess.run([tf.nn.sigmoid(pred_edges), tf.nn.sigmoid(pred_corners)], feed_dict=fd)
            print("sec/step :", time.time() - tt)


            # Save results
            imageio.imwrite(os.path.join(args.results, 'EM_test', filename + "_emap.jpg"), emap[0, :, :, 0])
            imageio.imwrite(os.path.join(args.results, 'CM_test', filename + "_emap.jpg"), cmap[0, :, :, 0])




def main():

    t = time.time()

    if not os.path.exists(os.path.join(args.results, 'EM_test')): os.makedirs(os.path.join(args.results, 'EM_test'))
    if not os.path.exists(os.path.join(args.results, 'CM_test')): os.makedirs(os.path.join(args.results, 'CM_test'))
    pred = predict(glob.glob(os.path.join(args.dataset, 'RGB') + '/*.jpg'))
    elapsed = time.time() - t
    print('Total time in seconds:', elapsed / 1)

    ## Give metrics
    P_e, R_e, Acc_e, f1_e, IoU_e = evaluate('edges')
    print('EDGES: IoU: ' + str('%.3f' % IoU_e) + '; Accuracy: ' + str('%.3f' % Acc_e) + '; Precision: ' + str(
        '%.3f' % P_e) + '; Recall: ' + str('%.3f' % R_e) + '; f1 score: ' + str('%.3f' % f1_e))
    P_c, R_c, Acc_c, f1_c, IoU_c = evaluate('corners')
    print('CORNERS: IoU: ' + str('%.3f' % IoU_c) + '; Accuracy: ' + str('%.3f' % Acc_c) + '; Precision: ' + str(
        '%.3f' % P_c) + '; Recall: ' + str('%.3f' % R_c) + '; f1 score: ' + str('%.3f' % f1_c))

    print("Final Accuracy accuracy EDGES:"+str('%.3f'%Acc_e)+";CORNERS:"+str('%.3f'%Acc_c))
    # latex format
    latex = [str('$%.3f$' % IoU_c) + " & " + str('$%.3f$' % Acc_c) + " & " + str('$%.3f$' % P_c) + " & " + str(
        '$%.3f$' % R_c) + " & " + str('$%.3f$' % f1_c)]
    print(latex)



if __name__ == '__main__':
    main()




