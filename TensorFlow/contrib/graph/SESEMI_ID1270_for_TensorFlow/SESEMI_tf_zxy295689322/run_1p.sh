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
#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}
current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 ${code_dir}/train_evaluate_asl2.py \
        --dataset=${data_dir} \
        --result=${result_dir} \
        --obs_url=${obs_url} \
        --data=cifar10 \
        --network=convnet  \
        --labels=1000
        #--gpu_id=1

