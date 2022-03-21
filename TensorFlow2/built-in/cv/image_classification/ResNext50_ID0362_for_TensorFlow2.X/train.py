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

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, model_index, save_every_n_epoch
from prepare_data import generate_datasets, load_and_preprocess_image
import math
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet, se_resnext
from models.RegNet import regnet
import time
from absl import flags
from absl import app
import npu_device
import ast

flags.DEFINE_string(name='precision_mode', default='allow_mix_precision',
                    help='allow_fp32_to_fp16/force_fp16/ '
                         'must_keep_origin_dtype/allow_mix_precision.')
flags.DEFINE_boolean(name='over_dump', default=False,
                     help='if or not over detection, default is False')
flags.DEFINE_boolean(name='data_dump_flag', default=False,
                     help='data dump flag, default is False')
flags.DEFINE_string(name='data_dump_step', default="10",
                    help='data dump step, default is 10')
flags.DEFINE_boolean(name='profiling', default=False,
                     help='if or not profiling for performance debug, default is False')
flags.DEFINE_string(name='profiling_dump_path', default="/home/data",
                    help='the path to save profiling data')
flags.DEFINE_string(name='over_dump_path', default="/home/data",
                    help='the path to save over dump data')
flags.DEFINE_string(name='data_dump_path', default="/home/data",
                    help='the path to save dump data')
flags.DEFINE_string(name='dataset_dir', default="/home/data",
                    help='the path to save dump data')
flags.DEFINE_boolean(name='use_mixlist', default=False,
                     help='use_mixlist flag, default is False')
flags.DEFINE_boolean(name='fusion_off_flag', default=False,
                     help='fusion_off flag, default is False')
flags.DEFINE_string(name='mixlist_file', default="ops_info.json",
                    help='mixlist file name, default is ops_info.json')
flags.DEFINE_string(name='fusion_off_file', default="fusion_switch.cfg",
                    help="fusion_off file name, default is fusion_switch.cfg")


def npu_config():
    FLAGS = flags.FLAGS
    npu_config = {}

    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "fp_point":"", \
                        "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode == 'allow_mix_precision':
        npu_device.global_options().modify_mixlist = FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file = FLAGS.fusion_off_file
    npu_device.open().as_default()


def get_model():
    if model_index == 0:
        return mobilenet_v1.MobileNetV1()
    elif model_index == 1:
        return mobilenet_v2.MobileNetV2()
    elif model_index == 2:
        return mobilenet_v3_large.MobileNetV3Large()
    elif model_index == 3:
        return mobilenet_v3_small.MobileNetV3Small()
    elif model_index == 4:
        return efficientnet.efficient_net_b0()
    elif model_index == 5:
        return efficientnet.efficient_net_b1()
    elif model_index == 6:
        return efficientnet.efficient_net_b2()
    elif model_index == 7:
        return efficientnet.efficient_net_b3()
    elif model_index == 8:
        return efficientnet.efficient_net_b4()
    elif model_index == 9:
        return efficientnet.efficient_net_b5()
    elif model_index == 10:
        return efficientnet.efficient_net_b6()
    elif model_index == 11:
        return efficientnet.efficient_net_b7()
    elif model_index == 12:
        return resnext.ResNeXt50()
    elif model_index == 13:
        return resnext.ResNeXt101()
    elif model_index == 14:
        return inception_v4.InceptionV4()
    elif model_index == 15:
        return inception_resnet_v1.InceptionResNetV1()
    elif model_index == 16:
        return inception_resnet_v2.InceptionResNetV2()
    elif model_index == 17:
        return se_resnet.se_resnet_50()
    elif model_index == 18:
        return se_resnet.se_resnet_101()
    elif model_index == 19:
        return se_resnet.se_resnet_152()
    elif model_index == 20:
        return squeezenet.SqueezeNet()
    elif model_index == 21:
        return densenet.densenet_121()
    elif model_index == 22:
        return densenet.densenet_169()
    elif model_index == 23:
        return densenet.densenet_201()
    elif model_index == 24:
        return densenet.densenet_264()
    elif model_index == 25:
        return shufflenet_v2.shufflenet_0_5x()
    elif model_index == 26:
        return shufflenet_v2.shufflenet_1_0x()
    elif model_index == 27:
        return shufflenet_v2.shufflenet_1_5x()
    elif model_index == 28:
        return shufflenet_v2.shufflenet_2_0x()
    elif model_index == 29:
        return resnet.resnet_18()
    elif model_index == 30:
        return resnet.resnet_34()
    elif model_index == 31:
        return resnet.resnet_50()
    elif model_index == 32:
        return resnet.resnet_101()
    elif model_index == 33:
        return resnet.resnet_152()
    elif model_index == 34:
        return se_resnext.SEResNeXt50()
    elif model_index == 35:
        return se_resnext.SEResNeXt101()
    elif model_index == 36:
        return regnet.RegNet()
    else:
        raise ValueError("The model_index does not exist.")


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


def main(_):
    # GPU settings
    npu_config()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    model = get_model()
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()
    # optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        train_acc_ave = 0
        train_acc_total = 0
        step = 0
        cost_time = 0
        for features in train_dataset:
            start_time = time.time()
            step += 1
            images, labels = process_features(features, data_augmentation=True)
            train_step(images, labels)
            train_acc_total += train_accuracy.result()
            cost_time += (time.time() - start_time)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}, perf: {:.5f}".format(epoch,
                                                                                                   EPOCHS,
                                                                                                   step,
                                                                                                   math.ceil(
                                                                                                       train_count / BATCH_SIZE),
                                                                                                   train_loss.result().numpy(),
                                                                                                   train_accuracy.result().numpy(),
                                                                                                   cost_time))
            cost_time = 0

        train_acc_ave = train_acc_total / (math.ceil(train_count / BATCH_SIZE))

        test_acc_ave = 0
        test_acc_total = 0
        test_count = 0

        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)
            test_acc_total += valid_accuracy.result()
            test_count += 1

        test_acc_ave = test_acc_total / test_count

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_acc_ave.numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  test_acc_ave.numpy()))
        # train_loss.reset_states()
        # train_accuracy.reset_states()
        # valid_loss.reset_states()
        # valid_accuracy.reset_states()

        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir + "epoch-{}".format(epoch), save_format='tf')

    # save weights
    model.save_weights(filepath=save_model_dir + "model", save_format='tf')
    tf.saved_model.save(model, './ckpt_npu')
    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    app.run(main)
