#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

#code_dir=${1}
#data_dir=${2}
#result_dir=${3}
#obs_url=${4}
#
#current_time=`date "+%Y-%m-%d-%H-%M-%S"`
#
#python3.7 ${code_dir}/train.py \
#        --dataset=${data_dir} \
#        --result=${result_dir} \
#        --obs_dir=${obs_url} \

#code_dir=$(cd "$(dirname "$0")"; cd ..; pwd)
#echo "===>>>Python boot file dir: ${code_dir}"
#code_dir=${1}
#data_dir=${2}
#result_dir=${3}
#obs_url=${4}
#bsrn_model=${5}
#scales_model=${6}
#clip_norm=${7}
#data_loader_model=${8}
#data_input_dir=${9}
#data_truth_dir=${10}

#代码目录，而非工作目录
code_dir=${1}
#Modelarts上的数据集存放路径
data_dir=${2}
#Modelarts上训练输出目录
result_dir=${3}
#OBS上路径
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python ${code_dir}/train.py \
        --data_input_path=${data_dir}/dataset/DIV2K/DIV2K_train_LR_bicubic\
        --data_truth_path=${data_dir}/dataset/DIV2K/DIV2K_train_HR \
        --train_path=${result_dir} \
        --obs_dir=${obs_url} \
        --chip='npu' \
        --platform='modelarts' \
        --model='bsrn' \
        --dataloader='div2k_loader' \
        --batch_size=8 \
        --max_steps=1000000 \
        --save_freq=10000 \
        --scales='4' # 2,3,4 for choosing
#       --bsrn_clip_norm=${clip_norm}