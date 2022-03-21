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

import os
import sys
import numpy as np
import argparse
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
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

parser = argparse.ArgumentParser(description='unsupervised training')
parser.add_argument('--dataset', type=str, required=True, default='',
                    help='Training dataset.')
parser.add_argument('--data_path', type=str, required=True, default='',
                    help='Directory contains required dataset.')

args = parser.parse_args()

# dataset
if args.dataset.upper() == 'DUKE':
    NUM_CLUSTER = 700
else:
    NUM_CLUSTER = 750
print(NUM_CLUSTER)
DATASET = args.data_path
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')

# learning
START = 1
END = 25
LAMBDA = 0.85
NUM_EPOCH = 20
BATCH_SIZE = 16

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# load data
unlabeled_images = []
with open(LIST, 'r') as f:
    for line in f:
        line = line.strip()
        img, lbl = line.split()
        img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        unlabeled_images.append(img)

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

# use GPU to calculate the similarity matrix
center_t = tf.placeholder(tf.float32, (None, None))
other_t = tf.placeholder(tf.float32, (None, None))
center_t_norm = tf.nn.l2_normalize(center_t, dim=1)
other_t_norm = tf.nn.l2_normalize(other_t, dim=1)
similarity = tf.matmul(center_t_norm, other_t_norm, transpose_a=False, transpose_b=True)

checkpoint = os.path.join(DATASET, "checkpoint")
checkpoint1 = os.path.join(checkpoint, "0.ckpt")
init_model = load_model(checkpoint1)
x1 = init_model.get_layer('avg_pool').output
x = x1
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
x = Dense(NUM_CLUSTER, activation='softmax', name='new_fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
init_model = Model(input=init_model.input, output=x1)
net = Model(input=init_model.input, output=x)
fc8_weights = net.get_layer('new_fc8').get_weights()
for layer in net.layers:
    layer.trainable = True
net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')

# iterate
for ckpt in range(START, END + 1):
    print("--------iterate ckpt----------")
    checkpoint = os.path.join(DATASET, "checkpoint")
    checkpoint1 = os.path.join(checkpoint, '%d.ckpt' % (ckpt - 1))
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
    similarities = sess.run(similarity, {center_t: centers, other_t: features})  # NUM_CLUSTER * num images

    # select reliable images
    reliable_image_idx = np.unique(np.argwhere(similarities > LAMBDA)[:, 1])
    print('ckpt %d: # reliable images %d' % (ckpt, len(reliable_image_idx)))
    sys.stdout.flush()
    images = np.array([unlabeled_images[i][0] for i in reliable_image_idx])
    labels = to_categorical([kmeans.labels_[i] for i in reliable_image_idx])

    # retrain: fine tune
    checkpoint = os.path.join(DATASET, "checkpoint")
    checkpoint1 = os.path.join(checkpoint, "0.ckpt")
    net.load_weights(checkpoint1, by_name=True)
    net.get_layer('new_fc8').set_weights(fc8_weights)

    net.fit_generator(datagen.flow(images, labels, batch_size=BATCH_SIZE), steps_per_epoch=len(images) / BATCH_SIZE + 1,
                      epochs=NUM_EPOCH)
    net.save(os.path.join(checkpoint, '%d.ckpt' % ckpt))
# tf.io.write_graph(sess.graph, './checkpoint', 'graph.pbtxt', as_text=True)
sess.close()
