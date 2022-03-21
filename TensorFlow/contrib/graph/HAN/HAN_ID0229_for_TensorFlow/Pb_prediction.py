# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import moxing as mox
# from models import HeteGAT_multi
from utils import process
import scipy.io as sio

#npu
from npu_bridge.estimator import npu_ops

batch_size = 1
hid_units = [8]
n_heads = [8, 1]
residual = False
nonlinearity = tf.nn.elu
# model = HeteGAT_multi

# data_url = '/home/wei/program/HAN_master_final/load_data/'
data_url = '/home/work/user-job-dir/code/load_data/'
data_file = 'ACM3025.mat'

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_dblp(path):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

# use adj_list as fea_list, have a try~
print("++++++++++++++++++++++++++++++++loading data+++++++++++++++++++++++++++++++++++++++++++++")
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp(data_url + data_file)
print("++++++++++++++++++++++++++++++++load data successly+++++++++++++++++++++++++++++++++++++++")

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

ftr_in_list_01 = 'input/ftr_in_0:0'
ftr_in_list_11 = 'input/ftr_in_1:0'
biases_list_01 = 'input/bias_in_0:0'
biases_list_11 = 'input/bias_in_1:0'
# attn_drop = 'input/attn_drop:0'
# ffd_drop = 'input/ffd_drop:0'

pb_file_path = '/home/work/user-job-dir/code/pb_model/HAN_final.pb'
# pb_file_path = '/home/wei/program/HAN_master_final/pb_model/frozen_model.pb'

with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')  # 导入计算图

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    ftr_in_list_0 = sess.graph.get_tensor_by_name(ftr_in_list_01)
    ftr_in_list_1 = sess.graph.get_tensor_by_name(ftr_in_list_11)
    biases_list_0 = sess.graph.get_tensor_by_name(biases_list_01)
    biases_list_1 = sess.graph.get_tensor_by_name(biases_list_11)

    # attn_drop_0 = sess.graph.get_tensor_by_name(attn_drop)
    # ffd_drop_0 = sess.graph.get_tensor_by_name(ffd_drop)

    logit = sess.graph.get_tensor_by_name('OutputNode:0')
    print(ftr_in_list_0)
    print(fea_list[0].shape)
    fd = {ftr_in_list_0: fea_list[0],
          ftr_in_list_1: fea_list[1],
          biases_list_0: biases_list[0],
          biases_list_1: biases_list[1]}
          # attn_drop: 0.0, ffd_drop: 0.0}

    logits1 = sess.run(logit, feed_dict=fd)

print(logits1)
logits_1 = np.array(logits1)

# logits_1.tofile("/home/wei/program/HAN_master_final/prediction/pb_prediction1.bin")

logits_1.tofile("/home/work/user-job-dir/code/pb_model_bin/pb_prediction1.bin")

print('done')

mox.file.copy_parallel('/home/work/user-job-dir/code/pb_model_bin', 'obs://hannet/MA-HAN-tf-Mingze-06-03-11-36/output_1')
