#!/bin/bash -eu
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
code_dir=${1}
DATA_DIR=${2}
result_dir=${3}
obs_url=${4}



python3.7 -u ${code_dir}/train_and_eval.py \
--obs_url=${obs_url} \
--task=supervised \
--dataset=imagenet \
--train_split=trainval \
--val_split=val \
--batch_size=128 \
--eval_batch_size=32 \
--workdir=${result_dir} \
#--dataset_dir= /cache/dataset \
--preprocessing=inception_preprocess \
--resize_size=224 \
--lr=0.1 \
--lr_scale_batch_size=256 \
--epochs=90 \
--warmup_epochs=5 \

