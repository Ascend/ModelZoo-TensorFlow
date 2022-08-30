# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
OUT_DIR=${3}

python3.7.5 -u $code_dir/generative.py \
--model_type=background \
--hidden_lstm_size=2000 \
--val_freq=500 \
--num_steps=300000 \
--mutation_rate=0.0 \
--reg_weight=0.0 \
--in_tr_data_dir=$DATA_DIR/before_2011_in_tr \
--in_val_data_dir=$DATA_DIR/between_2011-2016_in_val \
--ood_val_data_dir=$DATA_DIR/between_2011-2016_ood_val \
--out_dir=$OUT_DIR
