# !/bin/bash

export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/fwkacllib/ops/framework/built-in/tensorflow/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel:/usr/local/Ascend/fwkacllib/lib64/plugin/nnengine:/usr/local/Ascend/atc/lib64/plugin/opskernel:/usr/local/Ascend/atc/lib64/plugin/nnengine:/usr/local/Ascend/atc/lib64/stub:/usr/local/Ascend/acllib/lib64:/usr/local/python3.7/lib/:/usr/local/python3.7/lib/python3.7/site-packages/torch/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/atc/python/site-packages:/usr/local/Ascend/python/site-packages:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=/usr/local/Ascend/toolkit
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin:/usr/local/Ascend/fwkacllib/bin:/usr/local/Ascend/atc/bin:/usr/local/python3.7/bin/
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend
export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=600
