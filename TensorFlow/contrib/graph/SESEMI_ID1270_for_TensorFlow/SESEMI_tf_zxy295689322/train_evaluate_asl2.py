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
"""
Train and evaluate SESEMI architecture for supervised learning augmented
with self-supervised task of recognizing geometric transformations
defined as 90-degree rotations with horizontal and vertical flips.
"""

import os
os.environ['LD_PRELOAD'] = '/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so'

import npu_bridge.npu_init

# Python package imports
import argparse
# SESEMI package imports
from networks import convnet, wrn, nin
from dataset import svhn, cifar10, cifar100
from utils import global_contrast_normalize, zca_whitener
from utils import stratified_sample, gaussian_noise, datagen
from utils import LRScheduler, DenseEvaluator, compile_sesemi
# Keras package imports
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
from help_modelarts import obs_data2modelarts
import datetime
## Required parameters
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("result", "result", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("dataset", "dataset", "dataset path")
flags.DEFINE_string("obs_url", "obs://zhonglin-public/log", "obs result path, not need on gpu and apulis platform")

## Other parametersresult
flags.DEFINE_integer("labels", 1000, "number of classes for flowers datasets")
flags.DEFINE_string("network", "convnet", "Run on apulis/modelarts platform. \
    Modelarts Platform has some extra data copy operations")
flags.DEFINE_string("data", "cifar10", "Run on apulis/modelarts platform.\
     Modelarts Platform has some extra data copy operations")
flags.DEFINE_string("gpu_id", "0", "Run on apulis/modelarts platform.\
     Modelarts Platform has some extra data copy operations")
def open_sesemi():
    """
    open_sesemi.
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")
    print("===>>>dataset:{}".format(FLAGS.dataset))
    print("===>>>result:{}".format(FLAGS.result))
    print("===>>>data:{}".format(FLAGS.data))
    print("===>>>labels:{}".format(FLAGS.labels))
    print("===>>>network:{}".format(FLAGS.network))

    network = FLAGS.network
    data = FLAGS.data
    nb_labels = FLAGS.labels
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    
    arg2var = {'convnet': convnet,
               'wrn': wrn,
               'nin': nin,
               'svhn': svhn,
               'cifar10': cifar10,
               'cifar100': cifar100,}
    
    # Experiment- and dataset-dependent parameters.
    zca = True
    hflip = True
    epochs = 300 # between 100 to 300 epochs is good.
    batch_size = 16
    if data in {'svhn', 'cifar10'}:
        if data == 'svhn':
            zca = False
            hflip = False
        nb_classes = 10
    elif data == 'cifar100':
        nb_classes = 100
    else:
        raise ValueError('`dataset` must be "svhn", "cifar10", "cifar100".')
    super_dropout = 0.2
    in_network_dropout = 0.0
    if network == 'convnet' and data == 'svhn':
        super_dropout = 0.5
        in_network_dropout = 0.5

    # Prepare the dataset.
    (x_train, y_train), (x_test, y_test) = arg2var[data].load_data()
    
    x_test = global_contrast_normalize(x_test)
    x_train = global_contrast_normalize(x_train)
    
    if zca:
        zca_whiten = zca_whitener(x_train)
        x_train = zca_whiten(x_train)
        x_test = zca_whiten(x_test)

    x_test = x_test.reshape((len(x_test), 32, 32, 3))
    x_train = x_train.reshape((len(x_train), 32, 32, 3))
    
    if nb_labels in {50000, 73257}:
        x_labeled = x_train
        y_labeled = y_train
    else:
        labels_per_class = nb_labels // nb_classes
        sample_inds = stratified_sample(y_train, labels_per_class)
        x_labeled = x_train[sample_inds]
        y_labeled = y_train[sample_inds]

    y_labeled = to_categorical(y_labeled)
    
    # Training parameters.
    base_lr = 0.05
    lr_decay_power = 0.5
    input_shape = (32, 32, 3)
    max_iter = (len(x_labeled) // batch_size) * epochs

    # Compile the SESEMI model.
    sesemi_model, inference_model = compile_sesemi(
            arg2var[network], input_shape, nb_classes,
            base_lr, in_network_dropout, super_dropout
        )
    print(sesemi_model.summary())

    lr_poly_decay = LRScheduler(base_lr, max_iter, lr_decay_power)
    evaluate = DenseEvaluator(
            inference_model, (x_test, y_test), hflip, test_every=epochs)
    
    super_datagen = ImageDataGenerator(
            width_shift_range=[-2, -1, 0, 1, 2],
            height_shift_range=[-2, -1, 0, 1, 2],
            horizontal_flip=hflip,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )
    self_datagen = ImageDataGenerator(
            width_shift_range=[-2, -1, 0, 1, 2],
            height_shift_range=[-2, -1, 0, 1, 2],
            horizontal_flip=False,
            preprocessing_function=gaussian_noise,
            fill_mode='reflect',
        )

    super_data = super_datagen.flow(
            x_labeled, y_labeled, shuffle=True, batch_size=1, seed=None)
    self_data = self_datagen.flow(
            x_labeled, shuffle=True, batch_size=1, seed=None)
    train_data_loader = datagen(super_data, self_data, batch_size)
    
    # Fit the SESEMI model on mini-batches with data augmentation.
    print('Run configuration:')
    print('network=%s,' % network, 'dataset=%s,' % data, \
          'horizontal_flip=%s,' % hflip, 'ZCA=%s,' % zca, \
          'nb_epochs=%d,' % epochs, 'batch_size=%d,' % batch_size, \
          'nb_labels=%d,' % len(y_labeled), 'gpu_id=%s' % FLAGS.gpu_id)
    sesemi_model.fit_generator(train_data_loader,
                               epochs=epochs, verbose=1,
                               steps_per_epoch=len(x_labeled) // batch_size,
                               callbacks=npu_callbacks_append(callbacks_list=[lr_poly_decay, evaluate]),)


if __name__ == '__main__':
    """
    start function.
    """
    npu_keras_sess = set_keras_session_npu_config()
    open_sesemi()
    close_session(npu_keras_sess)


