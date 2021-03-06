#!/bin/bash
set -eux

dataset_path=${1}
ckpt=${2}
# checkpoint_path=${2:-mmnet-traindir}

python -W ignore offline_infer/convert_ckpt2pb.py \
    --num_classes 2 \
    --task_type matting \
    --output_name output/score \
    --output_type prob \
    --width 256 \
    --height 256 \
    --target_eval_shape 800 600 \
    --no-save_evaluation_image \
    --batch_size 1 \
    --dataset_path ${dataset_path} \
    --dataset_split_name test \
    --convert_to_pb \
    --preprocess_method preprocess_normalize \
    --no-use_fused_batchnorm \
    --valid_type loop \
    --max_outputs 1 \
    --augmentation_method resize_bilinear \
    --lambda_alpha_loss 1 \
    --lambda_comp_loss 1 \
    --lambda_grad_loss 1 \
    --lambda_kd_loss 1 \
    --lambda_aux_loss 1 \
    MMNetModel \
    --width_multiplier 1.0 \
    --ckpt_convert ${ckpt}
