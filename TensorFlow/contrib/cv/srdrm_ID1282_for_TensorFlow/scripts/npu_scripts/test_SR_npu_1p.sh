#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train_gpu
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}   # 当前工作路径
data_dir=${2}   # ModelArts上存放数据集的路径 /cache/dataset, 数据集路径
result_dir=${3}  # ModelArts上存放结果的路径 /cache/result, 输出结果路径
obs_url=${4}     # obs路径，modelarts运行时需要提供，其他情况可为空，裸机测试请修改对应参数paltform 为 linux
# test_epoch srdrm-gan 选 70, srdrm 选 52

python ${code_dir}/test_SR.py --test_mode 8x \
                     --chip npu \
                     --data_dir ${data_dir}/USR248/TEST/ \
                     --obs_dir ${obs_url} \
                     --model_name srdrm-gan \
                     --test_epoch 52 \
                     --result ${result_dir} \
                     --platform modelarts