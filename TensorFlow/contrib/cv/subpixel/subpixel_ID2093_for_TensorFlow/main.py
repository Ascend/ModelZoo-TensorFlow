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
from model import ESPCN
import numpy as np
import os
import time
import npu_bridge
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 15000, "Number of epoch")
flags.DEFINE_integer("image_size", 17, "The size of image input")
flags.DEFINE_integer("c_dim", 1, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer(
    "scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 14, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                    "Name of checkpoint directory")
flags.DEFINE_string("data_dir", "./", "Name of data directory")
flags.DEFINE_float("learning_rate", 1e-5, "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
log_dir = './log'  #add
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
if FLAGS.is_train:
    handler = logging.FileHandler(os.path.join(log_dir, now + "train_log.txt"))
else:
    handler = logging.FileHandler(os.path.join(log_dir, now + "test_log.txt"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(_):
    """
    函数入口
    """
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.gpu_options.allow_growth = True

    FLAGS.is_train = True
    with tf.Session(config=config) as sess:
        espcn = ESPCN(sess,
                      image_size=FLAGS.image_size,
                      is_train=FLAGS.is_train,
                      scale=FLAGS.scale,
                      c_dim=FLAGS.c_dim,
                      batch_size=FLAGS.batch_size,
                      test_img=FLAGS.test_img,
                      data_dir=FLAGS.data_dir,
                      logger=logger,
                      )
        espcn.train(FLAGS, logger)    #训练调用
    tf.reset_default_graph()

    FLAGS.is_train = False
    res = []
    bires = []
    set5 = ['baby_GT', 'bird_GT', 'butterfly_GT', 'head_GT', 'woman_GT']
    for index in range(len(set5)):
        with tf.Session(config=config) as sess:
            espcn = ESPCN(sess,
                          image_size=FLAGS.image_size,
                          is_train=FLAGS.is_train,
                          scale=FLAGS.scale,
                          c_dim=FLAGS.c_dim,
                          batch_size=FLAGS.batch_size,
                          test_img=FLAGS.test_img,
                          data_dir=FLAGS.data_dir,
                          logger=logger,
                          )

            temp, bi = espcn.test(FLAGS, logger, index)  #测试调用
            res.append(temp)
            bires.append(bi)
        tf.reset_default_graph()
    logger.info('res: {}, {}'.format(res, np.mean(res)))
    logger.info('bires: {}, {}'.format(bires, np.mean(bires)))


if __name__ == '__main__':
    tf.app.run()  # parse the command argument , the call the main function
