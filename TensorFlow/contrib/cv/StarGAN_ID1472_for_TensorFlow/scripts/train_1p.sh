#!/bin/bash

#export ASCEND_DEVICE_ID=0 # 根据硬件信息指定

phase="train"
image_root="../datasets/celeba/Img/img_align_celeba/"
metadata_path="../datasets/celeba/Anno/list_attr_celeba.txt"
batch_size=16
c_dim=5
selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"
training_iterations=2000
lr_update_step=10
num_step_decay=1000
output_dir="./logs/model_output_"`date +"%Y%m%d%H%M%S"`
summary_steps=10
save_steps=10
checkpoint_steps=10000

# for training
python3 main.py --phase="$phase" \
                  --image_root="$image_root" \
                  --metadata_path="$metadata_path" \
                  --batch_size="$batch_size" \
                  --c_dim="$c_dim" \
                  --selected_attrs="$selected_attrs" \
                  --training_iterations="$training_iterations" \
                  --lr_update_step="$lr_update_step" \
                  --num_step_decay="$num_step_decay" \
                  --output_dir="$output_dir" \
                  --summary_steps="$summary_steps" \
                  --save_steps="$save_steps" \
                  --checkpoint_steps="$checkpoint_steps"
