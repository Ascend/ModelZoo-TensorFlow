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
import scipy.io as sio
import numpy as np
from utils import process

data_url = 'load_data/ACM3025.mat'

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

adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp(path=data_url)

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


fd1 = fea_list[0]
fd1 = fd1.astype(np.float32)

fd2 = fea_list[1]
fd2 = fd2.astype(np.float32)

fd4 = biases_list[0]
fd4 = fd4.astype(np.float32)
fd5 = biases_list[1]
fd5 = fd5.astype(np.float32)

fd1.tofile("/home/wei/program/HAN_master_final/prediction/1.bin")
fd2.tofile("/home/wei/program/HAN_master_final/prediction/2.bin")
fd4.tofile("/home/wei/program/HAN_master_final/prediction/4.bin")
fd5.tofile("/home/wei/program/HAN_master_final/prediction/5.bin")


