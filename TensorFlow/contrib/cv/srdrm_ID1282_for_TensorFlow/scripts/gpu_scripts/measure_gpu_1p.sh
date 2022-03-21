#!/bin/bash
cd ../../

:<<!
以下命令执行推理(可以修改的参数如下所示)
--measure_mode 可选[2x,4x,8x]
--gt_dir   ground_truth的路径
--gen_dir_prefix 存储结果文件的路径，一般为checkpoints文件的上一级，在本地运行时一般为项目路径
--data_dir 测试数据集的路径
--model_name 要评估的模型
--platform 运行使用的平台
--test_epoch 与test_SR_gpu_1p.sh中的test_epoch值对应
!

data_dir=/mnt/data/wind/dataset/SRDRM/   # 数据集路径
obs_url=obs://srdrm/                     # obs路径, modelarts运行时需要，其他情况可不做修改
result_dir=/mnt/data/wind/SRDRM/         # 保存中间结果的路径，在本地运行时一般为项目路径

python measure.py --measure_mode 8x \
                  --chip gpu \
                  --gt_dir ${data_dir}/USR248/TEST/hr \
                  --obs_dir ${obs_url} \
                  --gen_dir_prefix ${result_dir} \
                  --test_epoch 52 \
                  --output ${result_dir} \
                  --result ${result_dir} \
                  --model_name srdrm-gan \
                  --platform linux
