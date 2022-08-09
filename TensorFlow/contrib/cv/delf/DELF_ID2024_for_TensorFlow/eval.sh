# coding=utf-8
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

## Extract query features
#python detect_to_retrieve/extract_query_features.py \
#  --delf_config_path detect_to_retrieve/delf_gld_config.pbtxt \
#  --dataset_file_path test/data/gnd_roxford5k.mat \
#  --images_dir test/data/oxford5k_images \
#  --output_features_dir test/data/oxford5k_features/query

python -u eval.py \
  --dataset_file_path test/data/gnd_roxford5k.mat \
  --images_dir test/data/oxford5k_tfrecords \
  --load_checkpoint \
  --logdir=result_v2
