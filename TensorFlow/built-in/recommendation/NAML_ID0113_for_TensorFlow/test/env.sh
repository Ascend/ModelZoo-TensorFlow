#!/bin/bash

# main env
export install_path=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/lib/:${install_path}/fwkacllib/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/fwkacllib/python/site-packages:${install_path}/tfplugin/python/site-packages
export PATH=$PATH:${install_path}/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=${install_path}/opp/
export SOC_VERSION=Ascend910
export EXPERIMENTAL_DYNAMIC_PARTITION=1 # 动态shape

#profiling env
export PROFILING_MODE=false  # true if need profiling
export PROFILING_OPTIONS="{\"output\":\"/autotest/profiling\",\"task_trace\":\"on\",\"training_trace\":\"on\",\"aicpu\":\"on\",\"fp_point\":\"user_encoder/time_distributed/news_encoder/subvert_encoder/embedding_2/embedding_lookup\",\"bp_point\":\"training/gradients/user_encoder/time_distributed/news_encoder/att_layer2_2/truediv_grad/RealDiv\",\"aic_metrics\":\"PipeUtilization\"}"

#debug env
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
#export DUMP_GE_GRAPH=2 # GE图
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1   #TF图
