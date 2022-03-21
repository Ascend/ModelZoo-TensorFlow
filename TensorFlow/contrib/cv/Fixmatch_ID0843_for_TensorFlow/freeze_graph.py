# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
CKPT_PATH = ''
INPUT_GRAPH = './pb_model/fixmatch.pb'
OUTPUT_GRAPH = './pb_albert_model/fixmatch_free_graph.pb'
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from absl import flags,app
from fixmatch import FixMatch
from libml import data,utils
FLAGS = flags.FLAGS

def main(argv):
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, FixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    ckpt = os.path.exists(FLAGS.CKPT_PATH)
    out_graph = os.path.exists(FLAGS.OUTPUT_GRAPH)
    pb_model = os.path.exists('./pb_model') 
    if not ckpt:
        print("ckpt folder is not exist")
    if not out_graph:
        os.makedirs(FLAGS.OUTPUT_GRAPH)
    if not pb_model:
        os.makedirs('./pb_model')

    with tf.compat.v1.Session() as sess:
        tf.io.write_graph(sess.graph_def, './pb_model', 'fixmatch.pb')
        freeze_graph.freeze_graph(
        input_graph=INPUT_GRAPH,
        input_saver='',
        input_binary=False,
        input_checkpoint=FLAGS.CKPT_PATH,
        output_node_names=FLAGS.OUTPUT_NODE_NAMES,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph= os.path.join(FLAGS.OUTPUT_GRAPH, "fixmatch.pb"),
        clear_devices=False,
        initializer_nodes='')


if __name__ == '__main__':
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_string('CKPT_PATH','','Select the checkpoint file place')
    flags.DEFINE_string('OUTPUT_GRAPH','','Select the graph file output place and PB_filename')
    flags.DEFINE_string('OUTPUT_NODE_NAMES','Softmax_2','Fill Output_node_name, default is Softmax_2')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.5@40-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
    # python freeze_graph.py --dataset=cifar10.5@40-1 \
    #                        --CKPT_PATH=./experiments/fixmatch/.../XXX.ckpt \
    #                        --OUTPUT_GRAPH=./pb_albert_model \
    #                        --OUTPUT_NODE_NAMES=Softmax_2