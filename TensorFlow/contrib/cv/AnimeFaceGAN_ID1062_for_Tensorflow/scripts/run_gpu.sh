#!/bin/bash
#set env
### GPU Platform command for train
# export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python train.py \
        --dataset=../dataset \
        --output=../output \
        --chip=gpu \
        --platform=linux \
        --num_classes=10 \
        --img_h=32 \
        --img_w=32 \
        --train_img_size=32 \
        --batch_size=64 \
        --train_itr=100000 \
        # --load_model
