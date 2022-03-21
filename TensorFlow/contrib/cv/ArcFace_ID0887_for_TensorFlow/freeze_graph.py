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
from tensorflow.python.tools import freeze_graph
import os

# from inception import inception_v1
slim = tf.contrib.slim
import argparse
from model import get_embd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--code_dir', help="""set code path""")
    parser.add_argument('--result', help="""set result path""")
    parser.add_argument('--config_path', type=str, help='path to config file', default='configs/config_ms1m_test1.yaml')
    parser.add_argument('--model_path', type=str, help='path to ckpt file',
                        default='/home/test_user02/Arcface/code/result/20210711-091916/checkpoints/ckpt-m-3093286')
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def main():
    args = parse_args()
    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 112, 112, 3], name="input")
    config = yaml.load(open(os.path.join(args.code_dir, args.config_path), 'r', encoding='utf-8'))
    embds, end_points = get_embd(inputs, False, False, config)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, args.result, 'model.pb')
        freeze_graph.freeze_graph(
            input_graph=os.path.join(args.result, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=args.model_path,
            output_node_names='embd_extractor/BatchNorm_1/Reshape_1',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(args.result, 'arcface_tf_310.pb'),  # graph outputs name
            clear_devices=False,
            initializer_nodes="")
    print("done")


if __name__ == '__main__':
    main()
