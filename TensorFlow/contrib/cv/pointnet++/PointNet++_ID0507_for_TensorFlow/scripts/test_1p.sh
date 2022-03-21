#!/bin/bash

#export ASCEND_DEVICE_ID=0 # 根据硬件信息指定

model="pointnet2_part_seg"
model_path="../part_seg/checkpoints/model.ckpt"
log_dir="log_eval"
num_point=2048
batch_size=8

python3 -u ../part_seg/evaluate.py --model="$model" \
                --model_path="$model_path" \
                --log_dir="$log_dir" \
                --num_point="$num_point" \
                --batch_size="$batch_size"
