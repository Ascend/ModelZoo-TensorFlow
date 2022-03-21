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
# Author: Tao Wu (taowu1@huawei.com)

# Train GCN with Cora-full dataset on GPU
python3.7 gcn/train.py \
    --device gpu \
    --cora_full \
    --take_subgraphs \
    --min_train_samples 20 \
    --min_valid_samples 30 \
    --valid_size 0 \
    --test_size 99999 \
    --num_epochs 2000 \
    --patience 200 \
    --hidden_dim 64 \
    --learning_rate 2e-3 \
    --keep_prob 0.2 \
    --l2_regularizer 2e-5 \
    --shuffle \
    --sparse_adj \
    --sparse_input \
    --pb_file constant_graph_corafull.pb

DATASET=corafull . scripts/prepare_inference_data.sh
