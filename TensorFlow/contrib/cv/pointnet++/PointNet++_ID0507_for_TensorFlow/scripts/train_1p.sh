#!/bin/bash

#export ASCEND_DEVICE_ID=0 # 根据硬件信息指定

model="pointnet2_part_seg"
log_dir="log"
num_point=2048
max_epoch=201
batch_size=8
learning_rate=0.001
momentum=0.9
optimizer="adam"
decay_step=200000
decay_rate=0.7

python3 -u ../part_seg/train.py --model="$model" \
                --log_dir="$log_dir" \
                --num_point="$num_point" \
                --max_epoch="$max_epoch" \
                --batch_size="$batch_size" \
                --learning_rate="$learning_rate" \
                --momentum="$momentum" \
                --optimizer="$optimizer" \
                --decay_step="$decay_step" \
                --decay_rate="$decay_rate"
