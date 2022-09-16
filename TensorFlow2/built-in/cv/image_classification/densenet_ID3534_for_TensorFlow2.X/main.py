# -*- coding: utf-8 -*-
#
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
import npu_device as npu
#npu.open().as_default()
import ast
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import densenet
from sklearn.model_selection import train_test_split
import os
import time

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help='data dir')
    parser.add_argument('--train_epochs', type=int, default=30, help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=100, help='Mini batch size')
    parser.add_argument('--log_steps', type=int, default=25, help='Steps to print log info')
    parser.add_argument('--steps_per_epoch', type=int, default=1440, help='Training epoch')
    # ===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str, help='precision mode')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                        help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,
                        help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str, help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str, help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str, help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                        help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                        help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,
                        help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,
                        help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def npu_config():
    if FLAGS.data_dump_flag:
        npu.global_options().dump_config.enable_dump = True
        npu.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu.global_options().dump_config.enable_dump_debug = True
        npu.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu.global_options().profiling_config.profiling_options = profiling_options
    npu.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu.global_options().auto_tune_mode="RL,GA"
    npu.open().as_default()

FLAGS = parse_args()
npu_config()

#######################
# Dimension of images #
#######################
img_width = 28
img_height = 28
channels = 1

######################
# Parms for learning #
######################
batch_size = FLAGS.batch_size
num_epochs = FLAGS.train_epochs
iterations = 1  # number of iterations
nb_augmentation = 2  # defines the number of additional augmentations of one image

####################
#       Data       #
####################
fashion_classes = {0: 'T-shirt/top',
                   1: 'Trouser',
                   2: 'Pullover',
                   3: 'Dress',
                   4: 'Coat',
                   5: 'Sandal',
                   6: 'Shirt',
                   7: 'Sneaker',
                   8: 'Bag',
                   9: 'Ankle boot'}

mnist_classes = [i for i in range(10)]
num_classes = 10

#### Loading data
# Train
train_fasion_mnist = tfds.as_numpy(tfds.load("fashion_mnist", data_dir=FLAGS.data_dir, download=False, split="train", batch_size=-1))
X_train, y_train = train_fasion_mnist["image"], train_fasion_mnist["label"]

# Test
test_fasion_mnist = tfds.as_numpy(tfds.load("fashion_mnist", data_dir=FLAGS.data_dir, download=False, split="test", batch_size=-1))
X_test, y_test = test_fasion_mnist["image"], test_fasion_mnist["label"]

print("Train Samples:", len(X_train))
print("Test Samples:", len(X_test))


#### Data augmentation (optional)
# Defines the options for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

def image_augmentation(image, nb_of_augmentation):
    '''
    Generates new images bei augmentation
    image : raw image
    nb_augmentation: number of augmentations
    images: array with new images
    '''
    images = []
    image = image.reshape(1, img_height, img_width, channels)
    i = 0
    for x_batch in datagen.flow(image, batch_size=1):
        images.append(x_batch)
        i += 1
        if i >= nb_of_augmentation:
            # interrupt augmentation
            break
    return images

#### Preprocess data
def preprocess_data(images, targets, use_augmentation=False, nb_of_augmentation=1):
    """
    images: raw image
    targets: target label
    use_augmentation: True if augmentation should be used
    nb_of_augmentation: If use_augmentation=True, number of augmentations
    """
    X = []
    y = []
    for x_, y_ in zip(images, targets):

        # scaling pixels between 0.0-1.0
        x_ = x_ / 255.

        # data Augmentation
        if use_augmentation:
            argu_img = image_augmentation(x_, nb_of_augmentation)
            for a in argu_img:
                X.append(a.reshape(img_height, img_width, channels))
                y.append(y_)

        X.append(x_)
        y.append(y_)
    print('*Preprocessing completed: %i samples\n' % len(X))
    return np.array(X), tf.keras.utils.to_categorical(y)

X_train_shaped, y_train_shaped = preprocess_data(
    X_train, y_train,
    use_augmentation=True,
    nb_of_augmentation=nb_augmentation
)

X_test_shaped, y_test_shaped = preprocess_data(X_test, y_test)

#### Model definition
# Usage of densenet
def create_model():
    model = densenet.DenseNet(input_shape=(28, 28, 1), nb_classes=10, depth=35, growth_rate=20,
                              dropout_rate=0.4, bottleneck=False, compression=0.9).build_model()
    optimizer=tf.optimizers.Adam()
    optimizer=npu.train.optimizer.NpuLossScaleOptimizer(optimizer, dynamic=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
create_model().summary()

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        self.start_time = None

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size
            print(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d'%(elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps),flush=True)
            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

# Run training
histories = []

for i in range(0, iterations):
    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train_shaped, y_train_shaped,
                                                          test_size=0.2, random_state=42)

    cnn = create_model()
    history = cnn.fit(
        X_train_, y_train_,
        batch_size=batch_size,
        epochs=num_epochs,
        steps_per_epoch=FLAGS.steps_per_epoch,
        verbose=2,
        validation_data=(X_val_, y_val_),
        callbacks=[
            TimeHistory(batch_size, FLAGS.log_steps)
        ]
    )

    histories.append(history.history)
    cnn.save_weights(filepath="checkpoint/tf_model", save_format="tf")