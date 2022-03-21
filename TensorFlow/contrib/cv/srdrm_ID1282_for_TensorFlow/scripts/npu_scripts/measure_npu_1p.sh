#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train_gpu
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}   # 当前工作路径
data_dir=${2}   # ModelArts上存放数据集的路径 /cache/dataset
result_dir=${3}  # ModelArts上存放结果的路径 /cache/result
obs_url=${4}     # obs://srdrm/

# --test_epoch 与test_SR_npu_1p.sh中的test_epoch值对应

python ${code_dir}/measure.py --measure_mode 8x \
                     --chip npu \
                     --gt_dir ${data_dir}/USR248/TEST/hr \
                     --obs_dir ${obs_url} \
                     --gen_dir_prefix ${result_dir} \
                     --test_epoch 52 \
                     --output ${result_dir} \
                     --result ${result_dir} \
                     --model_name srdrm-gan \
                     --platform modelarts
