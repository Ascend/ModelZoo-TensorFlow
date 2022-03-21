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

import npu_bridge
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import part_dataset
import provider
import tf_util
import part_dataset_all_normal


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='part_seg/pb_model/pointnet2.pb', help='Model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dataset_path', default='/home/test_user08/pointnet2/shapenetcore', help='dataset root')
parser.add_argument('--batch_size', default=1, help='batch size')
parser.add_argument('--num_class', default=50, help='class number')
FLAGS = parser.parse_args()

MODEL_PATH = os.path.join(ROOT_DIR, FLAGS.model_path)
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_CLASSES = FLAGS.num_class
DATA_PATH = FLAGS.dataset_path
BATCH_SIZE = FLAGS.batch_size

TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')


def get_model():
    # 昇腾AI处理器模型编译和优化配置
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    # 配置1： 选择在昇腾AI处理器上执行推理run on Ascend NPU
    custom_op.parameter_map["use_off_line"].b = True
    # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
    # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
    custom_op.parameter_map["graph_run_mode"].i = 0
    # 配置4：关闭remapping和MemoryOptimizer
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    graph = load_model(MODEL_PATH)
    pointclouds_pl = graph.get_tensor_by_name('input:0')
    output_tensor = graph.get_tensor_by_name('output:0')
    sess = tf.Session(config=config, graph=graph)
    ops = {'input': pointclouds_pl,
           'is_training_pl': tf.constant(False),
           'pred':output_tensor}
    return sess, ops


def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx6 array, return BxN pred '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0]//batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['input']: pc[i * batch_size:(i + 1) * batch_size, ...]}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size, ...] = batch_logits
    return np.argmax(logits, 2)


def load_model(model_file):
    """
    load frozen graph
    :param model_file:
    :return:
    """
    with tf.gfile.GFile(model_file, "rb") as gf:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(gf.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def get_batch(dataset, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps, normal, seg = dataset[i + start_idx]
        batch_data[i, :, 0:3] = ps
        batch_data[i, :, 3:6] = normal
        batch_label[i, :] = seg
    return batch_data, batch_label


if __name__ == '__main__':
    total_corrent = 0
    total_seen = 0
    for i in range(len(TEST_DATASET)):
        batch_data, seg = get_batch(TEST_DATASET, i, i+BATCH_SIZE)
        sess, ops = get_model()
        segp = inference(sess, ops, batch_data, batch_size=BATCH_SIZE)
        segp = segp.squeeze()
        correct = np.sum(segp == seg)
        seen = (BATCH_SIZE * NUM_POINT)
        total_corrent += correct
        total_seen += seen
        print("current data accuracy:", correct / float(seen))
    print("test accuracy:", total_corrent / float(total_seen))
