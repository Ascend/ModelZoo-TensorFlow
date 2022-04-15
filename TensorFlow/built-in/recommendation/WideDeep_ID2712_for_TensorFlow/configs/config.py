# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
BASE_DIR = './'

num_gpu = 1

batch_size = 131072
eval_batch_size = 32768
###
iterations_per_loop = 10
n_epoches = 2
n_epoches_8p = 4
#n_epoches = 1
#iterations_per_loop = 5
line_per_sample = 4096

#record_path = '/data/tf_record'
record_path = '/npu/traindata/ID2940_CarPeting_TF_WideDeep_TF/outbrain/tfrecords'
metadata_path = '/npu/traindata/ID2940_CarPeting_TF_WideDeep_TF/outbrain/tfrecords'
train_tag = 'train'
eval_tag = 'eval'
writer_path = './model/'
graph_path = './model/'

#timeline_analyse
timeline_npu_enable = True
timeline_initop_enable = True

#npu_switch
npu_enable = True

train_size = 59761827
#train_size = 1310720
test_size = 1048576 

#save model path
Base_path = './model/'

display_step = 100
