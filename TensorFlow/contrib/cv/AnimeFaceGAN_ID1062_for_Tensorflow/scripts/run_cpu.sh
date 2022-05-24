#!/bin/bash

python train.py \
        --dataset=../dataset \
        --output=../output \
        --chip=cpu \
        --platform=linux \
        --num_classes=10 \
        --img_h=32 \
        --img_w=32 \
        --train_img_size=32 \
        --batch_size=64 \
        --train_itr=100000 \
        # --load_model