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

# ATC environment variables
CANN_TOOLKIT_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
export PATH=${CANN_TOOLKIT_INSTALL_PATH}/atc/ccec_compiler/bin:${PATH}
export PATH=${CANN_TOOLKIT_INSTALL_PATH}/atc/bin:${PATH}
export PATH=${CANN_TOOLKIT_INSTALL_PATH}/toolkit/bin:${PATH}
export PATH=/usr/local/python3.7.5/bin:${PATH}
export LD_LIBRARY_PATH=${CANN_TOOLKIT_INSTALL_PATH}/atc/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${CANN_TOOLKIT_INSTALL_PATH}/atc/python/site-packages:${PYTHONPATH}
export PYTHONPATH=${CANN_TOOLKIT_INSTALL_PATH}/toolkit/python/site-packages:${PYTHONPATH}
export ASCEND_OPP_PATH=${CANN_TOOLKIT_INSTALL_PATH}/opp
export TOOLCHAIN_HOME=${CANN_TOOLKIT_INSTALL_PATH}/toolkit

# Ensure logs are produced to stdout
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
