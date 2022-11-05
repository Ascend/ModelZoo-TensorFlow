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

from __future__ import division, print_function, absolute_import
from operator import le

import os
import sys
import numpy as np
import argparse
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.initializers import RandomNormal
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from sklearn.cluster import KMeans
from npu_bridge.npu_init import *
# from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import time
RANK_SIZE = int(os.environ['RANK_SIZE'])
RANK_ID = int(os.environ['RANK_ID'])

if __name__ == '__main__':

    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    # parser
    parser = argparse.ArgumentParser(description='unsupervised training')
    parser.add_argument('--dataset', type=str, required=True, default='',
                        help='Training dataset.')
    parser.add_argument('--data_path', type=str, required=True, default='',
                        help='Directory contains required dataset.')
    parser.add_argument('--END', type=int, default=25, help='ckpt range')
    parser.add_argument('--end_step', type=int, default=0,
                        help='contrl steps_per_epoch')
    parser.add_argument('--NUM_EPOCH', type=int,
                        default=20, help='train epochs')
    parser.add_argument('--BATCH_SIZE', type=int,
                        default=16, help='BATCH_SIZE')
    parser.add_argument('--save_ckpt', type=str,
                        default='./save_ckpt', help='save ckpt')
    args = parser.parse_args()

    # dataset
    if args.dataset.upper() == 'DUKE':
        NUM_CLUSTER = 700
    else:
        NUM_CLUSTER = 750

    print(NUM_CLUSTER)
    DATASET = args.data_path
    save_ckpt = args.save_ckpt
    LIST = os.path.join(DATASET, 'train.list')
    TRAIN = os.path.join(DATASET, 'bounding_box_train')

    # learning
    START = 1
    END = args.END
    LAMBDA = 0.85
    NUM_EPOCH = args.NUM_EPOCH
    BATCH_SIZE = args.BATCH_SIZE

    # session
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(
        str(21 * 1024 * 1024 * 1024))
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(
        str(10 * 1024 * 1024 * 1024))
    # Allreduce并行
    if RANK_SIZE > 1:
        custom_op.parameter_map["hcom_parallel"].b = True

    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    # set_session(sess)
    # load data
    unlabeled_images = []
    with open(LIST, 'r') as f:
        print("--------openlist----------")
        for line in f:
            line = line.strip()
            img, lbl = line.split()
            img = image.load_img(os.path.join(TRAIN, img),
                                 target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            unlabeled_images.append(img)

    print("--------datagen----------")

    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=20,  # 0.
                                 width_shift_range=0.2,  # 0.
                                 height_shift_range=0.2,  # 0.
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None,
                                 data_format=K.image_data_format())

    # calculate the similarity matrix
    center_t = tf.placeholder(tf.float32, (None, None))
    other_t = tf.placeholder(tf.float32, (None, None))
    center_t_norm = tf.nn.l2_normalize(center_t, dim=1)
    other_t_norm = tf.nn.l2_normalize(other_t, dim=1)
    similarity = tf.matmul(center_t_norm, other_t_norm,
                           transpose_a=False, transpose_b=True)

    #checkpoint = os.path.join(DATASET, "checkpoint")
    checkpoint1 = os.path.join(save_ckpt, "0.ckpt")
    init_model = load_model(checkpoint1)
    x1 = init_model.get_layer('avg_pool').output
    x = x1
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLUSTER, activation='softmax', name='new_fc8',
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    init_model = Model(input=init_model.input, output=x1)
    net = Model(input=init_model.input, output=x)
    fc8_weights = net.get_layer('new_fc8').get_weights()
    for layer in net.layers:
        layer.trainable = True
    if RANK_SIZE > 1:
        net.compile(optimizer=npu_distributed_optimizer_wrapper(SGD(lr=0.001, momentum=0.9)),
                loss='categorical_crossentropy')
        callbacks = [NPUBroadcastGlobalVariablesCallback(0)]  # 变量进行广播
    else:
        net.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                loss='categorical_crossentropy')

     # 将images 和 labels 根据rankID分成8份
    if RANK_SIZE > 1:
         num_part = len(unlabeled_images) // RANK_SIZE
         unlabeled_images = unlabeled_images[RANK_ID * num_part:(RANK_ID + 1) * num_part]

    # iterate
    for ckpt in range(START, END + 1):
        print("--------iterate ckpt----------")
        #checkpoint = os.path.join(DATASET, "checkpoint")
        checkpoint1 = os.path.join(save_ckpt, '%d.ckpt' % (ckpt - 1))
        init_model.load_weights(checkpoint1, by_name=True)

        # extract features
        features = []
        for img in unlabeled_images:
            feature = init_model.predict(img)
            features.append(np.squeeze(feature))
        features = np.array(features)

        # clustering
        kmeans = KMeans(n_clusters=NUM_CLUSTER).fit(features)

        # select centers
        distances = kmeans.transform(features)  # num images * NUM_CLUSTER
        center_idx = np.argmin(distances, axis=0)
        centers = [features[i] for i in center_idx]

        # calculate similarity matrix
        # NUM_CLUSTER * num images
        similarities = sess.run(
            similarity, {center_t: centers, other_t: features})

        # calculate similarity matrixnet > LAMBDA)[:, 1])
        reliable_image_idx = np.unique(
            np.argwhere(similarities > LAMBDA)[:, 1])
        print('ckpt %d: # reliable images %d' %
              (ckpt, len(reliable_image_idx)))

        images = np.array([unlabeled_images[i][0] for i in reliable_image_idx])
        labels = to_categorical([kmeans.labels_[i]
                                for i in reliable_image_idx])

        # 将images 和 labels 根据rankID分成8份
        if RANK_SIZE > 1:
            images = images[:1440]
            labels = labels[:1440]

        # retrain: fine tune
        #checkpoint = os.path.join(DATASET, "checkpoint")
        checkpoint1 = os.path.join(save_ckpt, "0.ckpt")
        net.load_weights(checkpoint1, by_name=True)
        net.get_layer('new_fc8').set_weights(fc8_weights)
        if RANK_SIZE > 1:
            net.fit_generator(datagen.flow(images, labels, batch_size=BATCH_SIZE),
                              steps_per_epoch=len(
                                  images) // BATCH_SIZE - args.end_step,
                              epochs=NUM_EPOCH, callbacks=callbacks)
        else:
            net.fit_generator(datagen.flow(images, labels, batch_size=BATCH_SIZE),
                              steps_per_epoch=len(
                                  images) // BATCH_SIZE - args.end_step,
                              epochs=NUM_EPOCH)
        net.save(os.path.join(save_ckpt, '%d.ckpt' % ckpt))
    sess.close()

