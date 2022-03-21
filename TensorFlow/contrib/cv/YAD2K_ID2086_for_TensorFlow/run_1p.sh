#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train


code_dir=${1}
data_dir=${2}  # config.modelarts_data_dir 这个时候已经把obs中的数据什么的放到了 modelArts这个地址中了
result_dir=${3}  # config.modelarts_result_dir  同上
obs_url=${4} # config.train_url  s3://yolov2/yolov2forfen/output/V0003/


current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 ${code_dir}/train.py \
        --dataset=${data_dir} \
        --result=${result_dir} \
        --obs_dir=${obs_url}

