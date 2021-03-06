#!/usr/bin/env bash
#
# Copyright 2020 Huawei Technologies Co., Ltd
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

TASK_NAME="tnews"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export GLUE_DATA_DIR=$CURRENT_DIR/../../CLUEdataset/tnews/

# run task
cd $CURRENT_DIR
echo "Start predict pre process..."
python3 get_bert_tnews_infer_input.py \
  $GLUE_DATA_DIR \
  $GLUE_DATA_DIR/vocab.txt \
  $CURRENT_DIR/../datasets/

