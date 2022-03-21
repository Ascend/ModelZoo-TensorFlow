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


# ckpt文件在trained中，pb文件在pb_model中
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
import scipy.io as sio
import numpy as np
import moxing as mox
from models import HeteGAT_multi
from utils import process

#npu
from npu_bridge.estimator import npu_ops
batch_size = 1
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

ckpt_path = "/home/work/user-job-dir/code/trained/acm_allMP_multi_fea_.ckpt"
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
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

pb_path = "/home/work/user-job-dir/code/pb_model/"
pb_model = "model.pb"


########################################################
#这部分是用来生成输出节点的（原本网络中输出节点未命名）
########################################################
with tf.name_scope('input'):
    ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                  shape=(batch_size, nb_nodes, ft_size),
                                  name='ftr_in_{}'.format(i))
                   for i in range(len(fea_list))]
    bias_in_list = [tf.placeholder(dtype=tf.float32,
                                   shape=(batch_size, nb_nodes, nb_nodes),
                                   name='bias_in_{}'.format(i))
                    for i in range(len(biases_list))]
logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes,
                                                   bias_mat_list=bias_in_list,
                                                   hid_units=hid_units, n_heads=n_heads,
                                                   residual=residual, activation=nonlinearity
                                                   )
print(logits)
# 定义网络的输出节点
# predict_class = tf.identity(logits, name="output")


saver = tf.train.Saver()
with tf.Session() as sess:
    # model.pb文件将作为input_graph给到接下来的freeze_graph函数
    saver.restore(sess, ckpt_path)

    tf.train.write_graph(sess.graph_def, pb_path, pb_model)
    freeze_graph.freeze_graph(
        input_graph=pb_path + pb_model,  # 传入write_graph生成的模型文件
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
        output_node_names='OutputNode',  # 与定义的推理网络输出节点保持一致
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=pb_path+'HAN_final.pb',  # 改为需要生成的推理网络的名称
        clear_devices=False,
        initializer_nodes='')
print("done")
mox.file.copy_parallel('/home/work/user-job-dir/code/pb_model', 'obs://hannet/MA-HAN-tf-Mingze-06-03-11-36/pb_model')
