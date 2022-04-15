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


# -*- coding: utf-8 -*-

from model import CGAN
from utils import input_setup

import numpy as np
import tensorflow as tf
import sys
import pprint
import os
from cfg import make_config
from npu_bridge.npu_init import *


dataset_path = sys.argv[1]
result_path = sys.argv[2]
print(dataset_path)
print(result_path)

checkpoint_dir = os.path.join(result_path, "checkpoint")
sample_dir = os.path.join(result_path, "sample")
log_dir = os.path.join(result_path, "log")

flags = tf.app.flags
flags.DEFINE_integer("info_num", 0, "Number of epoch  step[0]") #性能看护需要的所见步数的控制变量
flags.DEFINE_integer("epoch", 15, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 132, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 120, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", sample_dir, "Name of sample directory [sample]")
flags.DEFINE_string("summary_dir", log_dir, "Name of log directory [log]")
flags.DEFINE_string("dataset",dataset_path,"dataset path")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("platform", "apulis", "Run on apulis/modelarts platform. Modelarts Platform has some extra data copy operations")

## The following params only useful on NPU chip mode
flags.DEFINE_boolean("npu_dump_data", False, "dump data for precision or not")
flags.DEFINE_boolean("npu_dump_graph", False, "dump graph or not")
flags.DEFINE_boolean("npu_profiling", False, "profiling for performance or not")
flags.DEFINE_boolean("npu_auto_tune", False, "auto tune or not. And you must set tune_bank_path param.")

FLAGS = flags.FLAGS
config = make_config(FLAGS)
pp = pprint.PrettyPrinter()
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")
    pp.pprint(FLAGS.__flags)
    for attr, flag_obj in sorted(FLAGS.__flags.items()):
        print("{} = {}".format(attr.lower(), flag_obj.value))

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    config_proto = tf.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    with tf.Session(config=config) as sess:
        srcnn = CGAN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  dataset=FLAGS.dataset
                 )
        srcnn.train(FLAGS)


    
if __name__ == '__main__':
    tf.app.run()
