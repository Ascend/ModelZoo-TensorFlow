#!/usr/bin/env bash

# Where the ImageNet2012 TFR is stored to. Replace this with yours
DATA_DIR=/home/sunshk/assemble/data/official_imagenet_kd

# Where the the checkpoint is saved to. Replace this with yours
MODEL_DIR=/home/sunshk/assemble/pretrained/Assemble-ResNet50/

ASCEND_DEVICE_ID=3 python main_classification.py \
--dataset_name=imagenet \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--preprocessing_type=imagenet_224_256 \
--batch_size=128 \
--mixup_type=1 \
--autoaugment_type=imagenet \
--resnet_version=2 \
--resnet_size=50 \
--use_sk_block=True \
--anti_alias_type=sconv \
--anti_alias_filter_size=3 \
--use_dropblock=True \
--num_gpus=1 \
--learning_rate_decay_type=cosine \
--weight_decay=1e-4 \
--base_learning_rate=0.05 \
--momentum=0.9 \
--lr_warmup_epochs=5 \
--zero_gamma=True \
--label_smoothing=0.1 \
--kd_temp=1 \
--dtype=fp16 \
--epochs_between_evals=10 \
--train_epochs=600

