#!/bin/bash

export ASCEND_DEVICE_ID=0 # 根据硬件信息指定
phase="test"
image_root="../datasets/celeba/images"
metadata_path="../datasets/celeba/list_attr_celeba.txt"
batch_size=1
c_dim=5
selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"
output_dir="../logs/model_output_"`date +"%Y%m%d%H%M%S"`
# which trained model will be used.
checkpoint="model-200000"

python3.7 ../main.py --phase="$phase" \
                  --image_root="$image_root" \
                  --metadata_path="$metadata_path" \
                  --batch_size="$batch_size" \
                  --c_dim="$c_dim" \
                  --selected_attrs="$selected_attrs" \
			      --output_dir="$output_dir" \
                  --checkpoint="$checkpoint"

