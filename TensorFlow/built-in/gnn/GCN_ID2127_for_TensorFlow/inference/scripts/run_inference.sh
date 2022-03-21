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

INFERENCE_DATA_DIR=inference/data/${DATASET}

INFERENCE_RESULTS_DIR=inference/results/${DATASET}
if [ ! -d "${INFERENCE_RESULTS_DIR}" ]; then
    mkdir -p ${INFERENCE_RESULTS_DIR}
fi

GCN_OFFLINE_MODEL=inference/model/om/constant_graph_${DATASET}.om
if [ ! -f "${GCN_OFFLINE_MODEL}" ]; then
    echo "[ERROR] Offline model file ${GCN_OFFLINE_MODEL} not found. Try running generate_om_${DATASET}.sh to fix it"
else
    # Inference execution
    python3 inference/src/main.py --model_path ${GCN_OFFLINE_MODEL} --input_path ${INFERENCE_DATA_DIR} --output_path ${INFERENCE_RESULTS_DIR} --sparse
fi
