#!/usr/bin/env bash

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

export ASCEND_GLOBAL_LOG_LEVEL=3

python ${code_dir}/main.py \
    --datasource=sinusoid --metatrain_iterations=60000 \
    --meta_batch_size=4 --update_lr=0.01 --norm=None --resume=True \
    --update_batch_size=10 --use_T=True --use_M=True --share_M=True \
    --logdir=${result_dir} --chip='npu' --platform='modelarts' --profiling=False \
    --obs_dir=${obs_url} 2>&1 | tee ${result_dir}/${current_time}_npu.log

