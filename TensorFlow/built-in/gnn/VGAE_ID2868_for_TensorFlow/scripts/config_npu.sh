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
# Author: Tao Wu

# CANN v5.0.2.x
export install_path=$HOME/Ascend/nnae/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}

export SOC_VERSION=Ascend910
export JOB_ID=10089
export ASCEND_DEVICE_ID=0
export ASCEND_GLOBAL_LOG_LEVEL=2

# For debugging only
# export DUMP_GE_GRAPH=1
# export DUMP_GRAPH_LEVEL=1
# export DUMP_GRAPH_PATH=./dumps
