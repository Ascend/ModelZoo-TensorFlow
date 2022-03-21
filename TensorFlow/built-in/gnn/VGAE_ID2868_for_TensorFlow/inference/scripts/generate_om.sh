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
# Author: Salli Moustafa 
#!/bin/bash

if [[ "x${DATASET}" == "x" || ("${DATASET}" != "cora" && "${DATASET}" != "citeseer" && "${DATASET}" != "pubmed") ]]; then
    echo "[WARNING] DATASET is not set. Available values: \"cora\", \"citeseer\", \"pubmed\". Default to \"cora\""
    DATASET=cora
fi

MODEL_PREFIX=constant_graph_${DATASET}
VGAE_CONSTANT_GRAPH=../model/pb/$MODEL_PREFIX.pb
OFFLINE_MODEL_OUTPUT_DIR=../model/om
if [ ! -d "${OFFLINE_MODEL_OUTPUT_DIR}" ]; then
    mkdir ${OFFLINE_MODEL_OUTPUT_DIR}
fi
OFFLINE_MODEL_OUTPUT=${OFFLINE_MODEL_OUTPUT_DIR}/$MODEL_PREFIX

# Input shape
if [ "${DATASET}" == "cora" ]; then
    FEATURES_SHAPE="Placeholders/features/indices:49216,2;Placeholders/features/values:49216;Placeholders/features/shape:2"
    ADJACENCY_SHAPE="Placeholders/adj/indices:11684,2;Placeholders/adj/values:11684;Placeholders/adj/shape:2"
    EDGES_POS="Placeholders/edges_pos:263,2"
    EDGES_NEG="Placeholders/edges_neg:263,2"
elif [ "${DATASET}" == "citeseer" ]; then
    FEATURES_SHAPE="Placeholders/features/indices:105165,2;Placeholders/features/values:105165;Placeholders/features/shape:2"
    ADJACENCY_SHAPE="Placeholders/adj/indices:11067,2;Placeholders/adj/values:11067;Placeholders/adj/shape:2"
    EDGES_POS="Placeholders/edges_pos:227,2"
    EDGES_NEG="Placeholders/edges_neg:227,2"
elif [ "${DATASET}" == "pubmed" ]; then
    FEATURES_SHAPE="Placeholders/features/indices:988031,2;Placeholders/features/values:988031;Placeholders/features/shape:2"
    ADJACENCY_SHAPE="Placeholders/adj/indices:95069,2;Placeholders/adj/values:95069;Placeholders/adj/shape:2"
    EDGES_POS="Placeholders/edges_pos:2216,2"
    EDGES_NEG="Placeholders/edges_neg:2216,2"
else
    echo "[ERROR] Unsupported dataset: ${DATASET}"
    exit -1
fi


INPUT_SHAPE="${FEATURES_SHAPE};${ADJACENCY_SHAPE};${EDGES_POS};${EDGES_NEG}"


# Output nodes
OUTPUT_NODES="Validation/preds_all:0;Validation/labels_all:0"
OUTPUT_TYPE="Validation/preds_all:0:FP16;Validation/labels_all:0:FP16"


# ATC vars
. cann_atc_vars.sh

# Model conversion
ASCEND_CHIP_NAME=`npu-smi info | head -7 | tail -1 | awk -F' ' '{print $3}'`
SOC_VERSION=Ascend${ASCEND_CHIP_NAME}

atc \
    --model=$VGAE_CONSTANT_GRAPH \
    --framework="3" \
    --soc_version="$SOC_VERSION" \
    --output=$OFFLINE_MODEL_OUTPUT \
    --log=info \
    --op_debug_level 2 \
    --display_model_info 1 \
    --input_shape="${INPUT_SHAPE}" \
    --input_format="ND" \
    --input_fp16_nodes="Placeholders/features/values;Placeholders/adj/values" \
    --out_nodes="${OUTPUT_NODES}" \
    --output_type="${OUTPUT_TYPE}"
