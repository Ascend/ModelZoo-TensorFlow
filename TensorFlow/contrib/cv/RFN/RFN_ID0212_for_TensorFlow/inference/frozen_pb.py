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


import sys
import argparse
import tensorflow as tf

sys.path.append("..") 
from models import RFN

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--pb", type=str, default="./")
args = parser.parse_args()

def convert_variables_to_constants(sess, args):
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))

    output_node_names = ['output'] ## output node names must be a list
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names
    )

    ## Finally we serialize and dump the output graph to the filesystem
    output_graph_path = "%s" % (args.pb_path)
    with tf.io.gfile.GFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def main(args):
    tf.reset_default_graph()
    
    # set inputs node, here you should use placeholder.
    inputs = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="input")
    # create inference graph
    with tf.variable_scope("RFN"):
        ret = RFN(rgba=inputs,
                  nf=64,
                  nb=24,
                  out_nc=3)
    uv = tf.identity(ret[0], name='output')

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        convert_variables_to_constants(sess, args)
        # freeze_model_graph(sess, args)

    print("Done")


main(args)