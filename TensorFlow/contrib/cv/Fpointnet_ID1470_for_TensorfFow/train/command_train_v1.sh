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
        --result=${result_dir} \
        --obs_dir=${obs_url} \
        --chip='npu' \
        --max_epoch=201 \
        --platform='modelarts' \
        --npu_dump_data=False \
        --npu_dump_graph=False \
        --npu_profiling=False \
        --npu_auto_tune=False  2>&1 | tee ${result_dir}/${current_time}_train_npu.log





##/bin/bash
#python train.py --gpu 0 --model frustum_pointnets_v1 --log_dir log --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5
