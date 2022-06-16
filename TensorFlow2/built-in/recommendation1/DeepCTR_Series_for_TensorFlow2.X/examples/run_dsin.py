#
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
#
#from npu_bridge.npu_init import *
import numpy as np
import tensorflow as tf

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat,get_feature_names
from deepctr.models import DSIN


def get_xy_fd(hash_flag=False):
    feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item', 3 + 1, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_0_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_0_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
            maxlen=4)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_1_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_1_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
            maxlen=4)]

    behavior_feature_list = ["item", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cateid = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    sess1_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [0, 0, 0, 0]])
    sess1_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [0, 0, 0, 0]])

    sess2_iid = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    sess2_cate_id = np.array([[1, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    sess_number = np.array([2, 1, 0])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'cate_id': cateid,
                    'sess_0_item': sess1_iid, 'sess_0_cate_id': sess1_cate_id, 'pay_score': score,
                    'sess_1_item': sess2_iid, 'sess_1_cate_id': sess2_cate_id, }

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    x["sess_length"] = sess_number
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    x, y, feature_columns, behavior_feature_list = get_xy_fd(True)

    model = DSIN(feature_columns, behavior_feature_list, sess_max_count=2,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, )

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)

