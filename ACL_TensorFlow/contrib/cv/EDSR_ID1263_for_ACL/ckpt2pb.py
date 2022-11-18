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
import run
import os
import argparse
from cfg import make_config
#from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.tools import freeze_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # gets rid of avx/fma warning

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("result", "result", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("dataset", "dataset", "dataset path")
flags.DEFINE_string("obs_dir", "obs://edsr/log", "obs result path, not need on gpu and apulis platform")

## Other parametersresult
flags.DEFINE_float("lr", 0.0001, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_blocks", 32, "number of resBlocks")
flags.DEFINE_integer("num_filters", 256, "number of filters")
flags.DEFINE_boolean("from_scrach", True, "train from scrach")
flags.DEFINE_integer("scale", 2, "Upscale an image with desired scale")
flags.DEFINE_integer("batch_size", 16, "batch size for one NPU")
flags.DEFINE_integer("train_steps", 100, "total epochs for training")
flags.DEFINE_integer("save_step", 5, "epochs for saving checkpoint")
flags.DEFINE_integer("decay_step", 500, "update the learning_rate value every decay_steps times")
flags.DEFINE_float("decay_rate", 0.9, "momentum used in optimizer")
flags.DEFINE_string("resume_path", None, "checkpoint path")
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("platform", "apulis",
                    "Run on apulis/modelarts platform. Modelarts Platform has some extra data copy operations")

## The following params only useful on NPU chip mode
flags.DEFINE_boolean("npu_dump_data", False, "dump data for precision or not")
flags.DEFINE_boolean("npu_dump_graph", False, "dump graph or not")
flags.DEFINE_boolean("npu_profiling", False, "profiling for performance or not")
flags.DEFINE_boolean("npu_auto_tune", False, "auto tune or not. And you must set tune_bank_path param.")

checkpoint = "./model/edsr_ckpt"

if __name__ == "__main__":

    # INIT
    scale = FLAGS.scale
    meanbgr = [103.1545782, 111.561547, 114.35629928]

    # Set checkpoint paths for different scales and models
    ckpt_path = ""
    if scale == 2:
        ckpt_path = os.path.join(FLAGS.result, "x2")
    elif scale == 3:
        ckpt_path = os.path.join(FLAGS.result, "x3")
    elif scale == 4:
        ckpt_path = os.path.join(FLAGS.result, "x3")
    else:
        print("No checkpoint directory. Choose scale 2, 3 or 4. Or add checkpoint directory for this scale.")
        exit()

    config = make_config(FLAGS)
    # Create run instance
    run = run.run(config, ckpt_path, FLAGS.scale, FLAGS.batch_size, FLAGS.train_steps, FLAGS.num_blocks,
                  FLAGS.num_filters, FLAGS.lr, FLAGS.from_scrach, meanbgr)


    tf.reset_default_graph()
    LR = tf.placeholder(tf.float32, shape=[None, 48, 48, 3], name="LR")
    out = run.test_model(LR)
    #out_nchw = tf.identity(out, name='out_nchw')
    with tf.Session(config=config) as sess:
    
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=checkpoint,
            output_node_names='NCHW_output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/edsr.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")