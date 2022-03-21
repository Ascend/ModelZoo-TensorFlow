#!/usr/bin/env bash

set -e


DATA_DIR=/home/sunshk/assemble/data/TFRecord_food101

# Where the the fine-tuned checkpoint is saved to. Replace this with yours
MODEL_DIR=/home/sunshk/assemble-multi/output_256a_b64_lr1e-3_wd1e-5

# Where the pretrained checkpoint is saved to. Replace this with yours
PRETRAINED_PATH=/home/sunshk/assemble/pretrained/Assemble-ResNet50/

ASCEND_DEVICE_ID=0 python main_classification.py \
--dataset_name=food101 \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--pretrained_model_checkpoint_path=${PRETRAINED_PATH} \
--num_gpus=1 \
--mixup_type=1 \
--autoaugment_type=good \
--label_smoothing=0.1 \
--resnet_version=2 \
--resnet_size=50 \
--anti_alias_filter_size=3 \
--anti_alias_type=sconv \
--use_sk_block=True \
--use_dropblock=True \
--dropblock_kp="0.9,0.7" \
--batch_size=64 \
--preprocessing_type=imagenet_224_256a \
--base_learning_rate=0.004 \
--learning_rate_decay_type=cosine \
--lr_warmup_epochs=5 \
--train_epochs=400 \
--bn_momentum=0.966 \
--weight_decay=0.0001 \
--keep_checkpoint_max=0 \
--ratio_fine_eval=1.0 \
--epochs_between_evals=1 \
--clean
