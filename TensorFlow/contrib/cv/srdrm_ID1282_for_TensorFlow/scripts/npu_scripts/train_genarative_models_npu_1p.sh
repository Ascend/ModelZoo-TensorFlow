#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train_gpu
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=${1}    # 项目路径
data_dir=${2}    # 数据集路径
result_dir=${3}  # 项目中间结果输出路径
obs_url=${4}     # obs路径，modelarts运行时需要提供，其他情况可为空，逻辑训练请修改对应参数paltform 为 linux

:<<!
以下命令执行训练srdrm模型(可以修改的参数如下所示)
--train_mode 可选[2x,4x,8x] 执行训练所用训练数据集
--num_epochs 模型训练的总epoch
--ckpt_interval 模型每隔多少epoch进行保存
--sample_interval 模型每隔多少step进行生成结果的采样输出
--batch_size
--data_path 数据集的路径
--start_epoch 从第几个epoch开始训练，默认从头开始
!
python ${code_dir}/train_genarative_models.py --train_mode 8x \
                                  --chip npu \
                                  --num_epochs 60 \
                                  --batch_size 2 \
                                  --data_path ${data_dir}/USR248/ \
                                  --output ${result_dir} \
                                  --obs_dir ${obs_url} \
                                  --start_epoch 0 \
                                  --platform modelarts