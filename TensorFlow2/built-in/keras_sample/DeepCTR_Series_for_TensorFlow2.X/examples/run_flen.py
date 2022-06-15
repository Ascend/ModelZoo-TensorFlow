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
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
from deepctr.feature_column import SparseFeat,get_feature_names
from deepctr.models import FLEN
import os 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./",
                        help='data path for train')
    parser.add_argument('--precision_mode', default='allow_fp32_to_fp16',
                        help='allow_fp32_to_fp16/force_fp16/ '
                             'must_keep_origin_dtype/allow_mix_precision.')
    parser.add_argument('--profiling', default=False,
                        help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data",
                        help='the path to save profiling data')
    args = parser.parse_args()

    session_config = tf.compat.v1.ConfigProto()
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    npu_keras_sess = set_keras_session_npu_config(config=session_config)

    data = pd.read_csv(os.path.join(args.data_dir, 'avazu_sample.txt'))
    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])

    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip',
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    field_info = dict(C14='user', C15='user', C16='user', C17='user',
                      C18='user', C19='user', C20='user', C21='user', C1='user',
                      banner_pos='context', site_id='context',
                      site_domain='context', site_category='context',
                      app_id='item', app_domain='item', app_category='item',
                      device_model='user', device_type='user',
                      device_conn_type='context', hour='context',
                      device_id='user'
                      )

    fixlen_feature_columns = [
        SparseFeat(name, vocabulary_size=data[name].max() + 1, embedding_dim=16, use_hash=False, dtype='int32',
                   group_name=field_info[name]) for name in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = FLEN(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=64, epochs=10, verbose=1, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=4)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    close_session(npu_keras_sess)


if __name__ == "__main__":
    main()
