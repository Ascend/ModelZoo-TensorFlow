#
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
"""
Title: Consistency training with supervision
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/04/13
Last modified: 2021/04/19
Description: Training with consistency regularization for robustness against data distribution shifts.
"""
"""
Deep learning models excel in many image recognition tasks when the data is independent
and identically distributed (i.i.d.). However, they can suffer from performance
degradation caused by subtle distribution shifts in the input data  (such as random
noise, contrast change, and blurring). So, naturally, there arises a question of
why. As discussed in [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/pdf/1906.08988.pdf)),
there's no reason for deep learning models to be robust against such shifts. Standard
model training procedures (such as standard image classification training workflows)
*don't* enable a model to learn beyond what's fed to it in the form of training data.

In this example, we will be training an image classification model enforcing a sense of
*consistency* inside it by doing the following:

* Train a standard image classification model.
* Train an _equal or larger_ model on a noisy version of the dataset (augmented using
[RandAugment](https://arxiv.org/abs/1909.13719)).
* To do this, we will first obtain predictions of the previous model on the clean images
of the dataset.
* We will then use these predictions and train the second model to match these
predictions on the noisy variant of the same images. This is identical to the workflow of
[*Knowledge Distillation*](https://keras.io/examples/vision/knowledge_distillation/) but
since the student model is equal or larger in size this process is also referred to as
***Self-Training***.

This overall training workflow finds its roots in works like
[FixMatch](https://arxiv.org/abs/2001.07685), [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848),
and [Noisy Student Training](https://arxiv.org/abs/1911.04252). Since this training
process encourages a model yield consistent predictions for clean as well as noisy
images, it's often referred to as *consistency training* or *training with consistency
regularization*. Although the example focuses on using consistency training to enhance
the robustness of models to common corruptions this example can also serve a template
for performing _weakly supervised learning_.

This example requires TensorFlow 2.4 or higher, as well as TensorFlow Hub and TensorFlow
Models, which can be installed using the following command:

"""

"""shell
pip install -q tf-models-official tensorflow-addons
"""

"""
## Imports and setup
"""
import imp
import npu_device
import time
import os
import ast
import numpy as np

from augment import RandAugment
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
import tensorflow as tf
import tensorflow_addons as tfa
# import matplotlib.pyplot as plt
import argparse

tf.random.set_seed(42)

"""
## Define hyperparameters
"""
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default="../cifar-10-batches-py/",
                        help="""directory to data""")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=5, type=int,
                        help="""epochs""")
    parser.add_argument("--log_steps", default=50, type=int,
                        help="TimeHis log Step.")
    parser.add_argument('--eval_static', dest="eval_static", type=ast.literal_eval,
                        help='the path to train data')
    #===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                        help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10",
                        help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                        help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                        help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()

def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
      npu_device.global_options().dump_config.enable_dump_debug = True
      npu_device.global_options().dump_config.dump_path = args.over_dump_path
      npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
      npu_device.global_options().profiling_config.enable_profiling = True
      profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                          "training_trace":"on", \
                          "task_trace":"on", \
                          "aicpu":"on", \
                          "aic_metrics":"PipeUtilization",\
                          "fp_point":"", \
                          "bp_point":""}'
      npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode = args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  npu_device.open().as_default()
#===============================NPU Migration=========================================

print('npu_device loaded')
npu_config()

data_path = args.data_dir

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

CROP_TO = 72
RESIZE_TO = 96

"""
## Load the CIFAR-10 dataset
"""

def load_data(data_path):
    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(data_path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
            y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(data_path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data(data_path)

val_samples = 49500
new_train_x, new_y_train = x_train[: val_samples + 1], y_train[: val_samples + 1]
val_x, val_y = x_train[val_samples:], y_train[val_samples:]

"""
## Create TensorFlow `Dataset` objects
"""

# Initialize `RandAugment` object with 2 layers of
# augmentation transforms and strength of 9.
augmenter = RandAugment(num_layers=2, magnitude=9)

"""
For training the teacher model, we will only be using two geometric augmentation
transforms: random horizontal flip and random crop.
"""


def preprocess_train(image, label, noisy=True):
    image = tf.image.random_flip_left_right(image)
    # We first resize the original image to a larger dimension
    # and then we take random crops from it.
    image = tf.image.resize(image, [RESIZE_TO, RESIZE_TO])
    image = tf.image.random_crop(image, [CROP_TO, CROP_TO, 3])
    if noisy:
        image = augmenter.distort(image)
    return image, label


def preprocess_test(image, label):
    image = tf.image.resize(image, [CROP_TO, CROP_TO])
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((new_train_x, new_y_train))
validation_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

"""
We make sure `train_clean_ds` and `train_noisy_ds` are shuffled using the *same* seed to
ensure their orders are exactly the same. This will be helpful during training the
student model.
"""
if args.eval_static:

    # This dataset will be used to train the first model.
    train_clean_ds = (
        train_ds.shuffle(BATCH_SIZE * 10, seed=42)
        .map(lambda x, y: (preprocess_train(x, y, noisy=False)), num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )

    # This prepares the `Dataset` object to use RandAugment.
    train_noisy_ds = (
        train_ds.shuffle(BATCH_SIZE * 10, seed=42)
        .map(preprocess_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )

    validation_ds = (
        validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )

    test_ds = (
        test_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )
else:
    # This dataset will be used to train the first model.
    train_clean_ds = (
        train_ds.shuffle(BATCH_SIZE * 10, seed=42)
        .map(lambda x, y: (preprocess_train(x, y, noisy=False)), num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # This prepares the `Dataset` object to use RandAugment.
    train_noisy_ds = (
        train_ds.shuffle(BATCH_SIZE * 10, seed=42)
        .map(preprocess_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    validation_ds = (
        validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    test_ds = (
        test_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
# This dataset will be used to train the second model.
consistency_training_ds = tf.data.Dataset.zip((train_clean_ds, train_noisy_ds))

"""
## Visualize the datasets
"""

# sample_images, sample_labels = next(iter(train_clean_ds))
# plt.figure(figsize=(10, 10))
# for i, image in enumerate(sample_images[:9]):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image.numpy().astype("int"))
#     plt.axis("off")

# sample_images, sample_labels = next(iter(train_noisy_ds))
# plt.figure(figsize=(10, 10))
# for i, image in enumerate(sample_images[:9]):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image.numpy().astype("int"))
#     plt.axis("off")

"""
## Define a model building utility function

We now define our model building utility. Our model is based on the [ResNet50V2 architecture](https://arxiv.org/abs/1603.05027).
"""


def get_training_model(num_classes=10):
    resnet50_v2 = tf.keras.applications.ResNet50V2(
        weights=None, include_top=False, input_shape=(CROP_TO, CROP_TO, 3),
    )
    model = tf.keras.Sequential(
        [
            layers.Input((CROP_TO, CROP_TO, 3)),
            layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            resnet50_v2,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes),
        ]
    )
    return model


"""
In the interest of reproducibility, we serialize the initial random weights of the
teacher  network.
"""

initial_teacher_model = get_training_model()
initial_teacher_model.save_weights("initial_teacher_model.h5")

"""
## Train the teacher model

As noted in Noisy Student Training, if the teacher model is trained with *geometric
ensembling* and when the student model is forced to mimic that, it leads to better
performance. The original work uses [Stochastic Depth](https://arxiv.org/abs/1603.09382)
and [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) to bring in the ensembling
part but for this example, we will use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
(SWA) which also resembles geometric ensembling.
"""

# Define the callbacks.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        #self.opt = optimizer
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

# Initialize SWA from tf-hub.
#SWA = tfa.optimizers.SWA
from My_SWA import My_SWA

# Compile and train the teacher model.
teacher_model = get_training_model()
teacher_model.load_weights("initial_teacher_model.h5")
teacher_model.compile(
    # Notice that we are wrapping our optimizer within SWA
    optimizer=My_SWA(tf.keras.optimizers.Adam()),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = teacher_model.fit(
    train_clean_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    callbacks=[reduce_lr, TimeHistory(args.batch_size,args.log_steps)],
    verbose=2,
)

#Evaluate the teacher model on the test set.
_, acc = teacher_model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc}")

"""
## Define a self-training utility

For this part, we will borrow the `Distiller` class from [this Keras Example](https://keras.io/examples/vision/knowledge_distillation/).
"""

# Majority of the code is taken from:
# https://keras.io/examples/vision/knowledge_distillation/
class SelfTrainer(tf.keras.Model):
    def __init__(self, student, teacher):
        super(SelfTrainer, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(
        self, optimizer, metrics, student_loss_fn, distillation_loss_fn, temperature=3,
    ):
        super(SelfTrainer, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def train_step(self, data):
        # Since our dataset is a zip of two independent datasets,
        # after initially parsing them, we segregate the
        # respective images and labels next.
        clean_ds, noisy_ds = data
        clean_images, _ = clean_ds
        noisy_images, y = noisy_ds

        # Forward pass of teacher
        teacher_predictions = self.teacher(clean_images, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(noisy_images, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            total_loss = (student_loss + distillation_loss) / 2

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`
        self.compiled_metrics.update_state(
            y, tf.nn.softmax(student_predictions, axis=1)
        )

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"total_loss": total_loss})
        return results

    def test_step(self, data):
        # During inference, we only pass a dataset consisting images and labels.
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Update the metrics
        self.compiled_metrics.update_state(y, tf.nn.softmax(y_prediction, axis=1))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        return results


"""
The only difference in this implementation is the way loss is being calculated. **Instead
of weighted the distillation loss and student loss differently we are taking their
average following Noisy Student Training**.
"""

"""
## Train the student model
"""

# Define the callbacks.
# We are using a larger decay factor to stabilize the training.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=3, factor=0.5, monitor="val_accuracy"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, monitor="val_accuracy"
)

# Compile and train the student model.
self_trainer = SelfTrainer(student=get_training_model(), teacher=teacher_model)
self_trainer.compile(
    # Notice we are *not* using SWA here.
    optimizer="adam",
    metrics=["accuracy"],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    temperature=10,
)
history = self_trainer.fit(
    consistency_training_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    callbacks=[reduce_lr, TimeHistory(args.batch_size,args.log_steps)],
    verbose=2
)

# Evaluate the student model.
acc = self_trainer.evaluate(test_ds, verbose=0)
print(f"Test accuracy from student model: {acc}")

"""
## Assess the robustness of the models

A standard benchmark of assessing the robustness of vision models is to record their
performance on corrupted datasets like ImageNet-C and CIFAR-10-C both of which were
proposed in [Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations](https://arxiv.org/abs/1903.12261). For this example, we will be using the
CIFAR-10-C dataset which has 19 different corruptions on 5 different severity levels. To
assess the robustness of the models on this dataset, we will do the following:

* Run the pre-trained models on the highest level of severities and obtain the top-1
accuracies.
* Compute the mean top-1 accuracy.

For the purpose of this example, we won't be going through these steps. This is why we
trained the models for only 5 epochs. You can check out [this
repository](https://github.com/sayakpaul/Consistency-Training-with-Supervision) that
demonstrates the full-scale training experiments and also the aforementioned assessment.
The figure below presents an executive summary of that assessment:

![](https://i.ibb.co/HBJkM9R/image.png)

**Mean Top-1** results stand for the CIFAR-10-C dataset and **Test Top-1** results stand
for the CIFAR-10 test set. It's clear that consistency training has an advantage on not
only enhancing the model robustness but also on improving the standard test performance.
"""