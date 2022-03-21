#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 ${code_dir}/mnasnet_main.py \
        --data_dir=${data_dir} \
        --model_dir=${result_dir} \
        --obs_dir=${obs_url} \
        --model_name='mixnet-s' \
        --npu_dump_data=True \
        --npu_profiling=True \
        --npu_dump_graph=False \
        --npu_auto_tune=False  2>&1 | tee ${result_dir}/${current_time}_train_npu.log