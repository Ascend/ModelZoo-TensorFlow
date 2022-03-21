# -*- coding:utf-8 -*-
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


import tensorflow as tf
from model import stargan
from npu_bridge.npu_init import *


def main(_):
    '''
    An implementation of StarGAN using TensorFlow (work in progress).

    :param _:
    :return:
    '''
    tf_flags = tf.app.flags.FLAGS
    # npu config
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess:
            # when use queue to load data, not use with to define sess
            train_model = stargan.StarGAN(sess, tf_flags)
            train_model.train(tf_flags.image_root, tf_flags.metadata_path, tf_flags.training_iterations,
                              tf_flags.summary_steps, tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = stargan.StarGAN(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)

            # test for images
            test_model.test(tf_flags.image_root, tf_flags.metadata_path)
            print("Saved testing files.")


if __name__ == '__main__':
    # Directories.
    tf.app.flags.DEFINE_string("output_dir", "model_output",
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("image_root", "./datasets/CelebA_nocrop/images",
                               "the root path of images.")
    tf.app.flags.DEFINE_string("metadata_path", "./datasets/list_attr_celeba.txt",
                               "the path of metadata.")

    # Model configuration.
    tf.app.flags.DEFINE_string("phase", "train",
                               "model phase: train/test.")
    tf.app.flags.DEFINE_integer("image_size", 128,
                                "image resolution.")
    tf.app.flags.DEFINE_integer("g_conv_dim", 64,
                                "number of conv filters in the first layer of G.")
    tf.app.flags.DEFINE_integer("d_conv_dim", 64,
                                "number of conv filters in the first layer of D.")
    tf.app.flags.DEFINE_integer("g_repeat_num", 6,
                                "number of strided conv layers in G.")
    tf.app.flags.DEFINE_integer("d_repeat_num", 6,
                                "number of strided conv layers in D.")
    tf.app.flags.DEFINE_float("lambda_cls", 1.,
                              "scale cls loss.")
    tf.app.flags.DEFINE_float("lambda_rec", 10.,
                              "scale G rec loss.")
    tf.app.flags.DEFINE_float("lambda_gp", 10.,
                              "scale gradient penalty loss.")

    # Training configuration
    tf.app.flags.DEFINE_integer("batch_size", 16,
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("c_dim", 5,
                                "the dimension of condition.")
    tf.app.flags.DEFINE_string("selected_attrs", 'Black_Hair Blond_Hair Brown_Hair Male Young',
                               "selected attributes for the CelebA dataset")
    tf.app.flags.DEFINE_integer("d_train_repeat", 5,
                                "the frequency of training Discriminator network.")
    tf.app.flags.DEFINE_float("init_learning_rate", 0.0001,
                              "learning_rate update step.")
    tf.app.flags.DEFINE_float("lr_update_step", 10,
                              "learning_rate update step.")
    tf.app.flags.DEFINE_float("num_step_decay", 1000,
                              "step for starting decay learning_rate.")
    tf.app.flags.DEFINE_float("beta1", 0.5,
                              "beta1 for Adam optimizer.")
    tf.app.flags.DEFINE_float("beta2", 0.999,
                              "beta2 for Adam optimizer.")
    tf.app.flags.DEFINE_integer("training_iterations", 200000,
                                "number of total iterations for training D.")

    # Step size.
    tf.app.flags.DEFINE_integer("summary_steps", 100,
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000,
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 500,
                                "save sample period.")

    # Test configuration.
    tf.app.flags.DEFINE_string("checkpoint", None,
                               "checkpoint name for restoring.")
    tf.app.flags.DEFINE_string("test_concatenate", "true",
                               "save the results with concatenated.")
    tf.app.run(main=main)
