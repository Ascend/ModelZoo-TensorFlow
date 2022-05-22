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

## build train dataset
#nohup /usr/local/python3.7.5/bin/python3.7 -u build_image_dataset.py \
#  --train_clean_csv_path=GLDv2_train_cleaned.csv \
#  --train_directory=/root/projects/google-landmark/train/*/*/*/ \
#  --output_directory=/root/projects/google-landmark/tfrecord/ \
#  --num_shards=128 \
#  --generate_train_validation_splits \
#  --validation_split_size=0.2 > log_build &

# build train dataset
python -u build_test_dataset.py \
  --dataset_file_path test/data/gnd_roxford5k.mat \
  --images_dir test/data/oxford5k_images \
  --tfrecords_dir test/data/oxford5k_tfrecords
