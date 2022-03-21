#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)



#代码目录，而非工作目录
code_dir=${1}
#Modelarts上的数据集存放路径
data_dir=${2}
#Modelarts上训练输出目录
result_dir=${3}
#OBS上路径
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`



python ${code_dir}/validate_bsrn.py \
        --dataloader=basic_loader \
        --data_input_path=${data_dir}/dataset/BSD100/LR \
        --data_truth_path=${data_dir}/dataset/BSD100/SR \
        --restore_path=${data_dir}/result/model.ckpt-100 \
        --obs_dir=${obs_url} \
        --model=bsrn \
        --scales=4 \
        --save_path=${data_dir}/result/result-pictures \
        --train_path=${result_dir} \
        --chip='npu' \
        --platform='modelarts'
