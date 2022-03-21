#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
###python3.7 ${code_dir}/train.py ${code_dir}/config/sphere64_casia.py
###python3.7 ${code_dir}/train.py ${code_dir}config/sphere64_casia.py 2>&1 | tee ${result_dir}/${current_time}_train_npu.log
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 /home/ma-user/modelarts/user-job-dir/code/train.py config/sphere64_casia.py --code_dir ${code_dir} --obs_url ${obs_url}
###python3.7 ${code_dir}/train.py config/sphere64_casia.py --code_dir ${code_dir} --obs_url ${obs_url}