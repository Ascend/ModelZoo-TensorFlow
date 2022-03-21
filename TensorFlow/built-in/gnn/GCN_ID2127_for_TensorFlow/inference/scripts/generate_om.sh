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
# Author: Salli Moustafa (salli.moustafa@huawei.com)
#!/bin/bash

if [[ "x${DATASET}" == "x" || ("${DATASET}" != "cora" && "${DATASET}" != "corafull") ]]; then
    echo "[WARNING] DATASET is not set. Available values: \"cora\", \"corafull\". Default to \"cora\""
    DATASET=cora
fi

MODEL_PREFIX=constant_graph_${DATASET}
GCN_CONSTANT_GRAPH=../model/pb/$MODEL_PREFIX.pb
OFFLINE_MODEL_OUTPUT_DIR=../model/om
if [ ! -d "${OFFLINE_MODEL_OUTPUT_DIR}" ]; then
    mkdir ${OFFLINE_MODEL_OUTPUT_DIR}
fi
OFFLINE_MODEL_OUTPUT=${OFFLINE_MODEL_OUTPUT_DIR}/$MODEL_PREFIX

# Input shape
if [ "${DATASET}" == "cora" ]; then
    if [ "${SPARSE}" == "1" ]; then
        FEATURES_SHAPE="Placeholders/X_data:45487;Placeholders/X_idx:45487,2"
        ADJACENCY_SHAPE="Placeholders/A_data:12623;Placeholders/A_idx:12623,2"
    else
        FEATURES_SHAPE="Placeholders/X_data:2485,1433"
        ADJACENCY_SHAPE="Placeholders/A_data:2485,2485"
    fi
    MASK_SHAPE="Placeholders/mask_valid:1000"
elif [ "${DATASET}" == "corafull" ]; then
    if [ "${SPARSE}" == "1" ]; then
        FEATURES_SHAPE="Placeholders/X_data:1071300;Placeholders/X_idx:1071300,2"
        ADJACENCY_SHAPE="Placeholders/A_data:143560;Placeholders/A_idx:143560,2"
    else
        FEATURES_SHAPE="Placeholders/X_data:18712,8710"
        ADJACENCY_SHAPE="Placeholders/A_data:18712,18712"
    fi
    MASK_SHAPE="Placeholders/mask_valid:15362"
else
    echo "[ERROR] Unsupported dataset: ${DATASET}"
    exit -1
fi


INPUT_SHAPE="${FEATURES_SHAPE};${ADJACENCY_SHAPE};${MASK_SHAPE}"


# Output node
OUTPUT_NODE="masked_logits_2:0"


# ATC vars
. cann_atc_vars.sh

# Model conversion
atc --model=$GCN_CONSTANT_GRAPH \
    --framework="3" \
    --soc_version="Ascend310" \
    --output=$OFFLINE_MODEL_OUTPUT \
    --log=warning \
    --input_shape="${INPUT_SHAPE}" \
    --input_format="ND" \
    --input_fp16_nodes="Placeholders/X_data;Placeholders/A_data" \
    --out_nodes="${OUTPUT_NODE}" \
    --output_type="${OUTPUT_NODE}:FP16"
