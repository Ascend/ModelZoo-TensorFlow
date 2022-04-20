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

python3.7 ${code_dir}/train.py \
        --dataset=${data_dir} \
        --output=${result_dir} \
        --obs_dir=${obs_url} \
        --chip=npu \
        --platform=modelarts \
        --num_classes=10 \
        --img_h=32 \
        --img_w=32 \
        --train_img_size=32 \
        --train_itr=100000 \
        --batch_size=64 \
#        --use_fp16 \
#        --profiling \
#        --load_model \
