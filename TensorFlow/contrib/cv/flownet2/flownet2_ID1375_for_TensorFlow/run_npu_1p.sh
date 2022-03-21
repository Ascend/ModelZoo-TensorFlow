#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

checkpoint_dir=./checkpoints

code_dir=$(cd "$(dirname "$0")"; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3 ${code_dir}/flownet2_train.py \
        --dataset='./data/sintel' \
        --result='./log/flownet2' \
        --train_file='./data/sintel/train.txt' \
        --val_file='./data/sintel/val.txt' \
        --batch_size=6 \
        --save_step=10 \
        --image_size=436,1024 \
        --pretrained=${checkpoint_dir}/FlowNet2/flownet-2.ckpt-0 \
        --resume_path='./log/flownet2' \
        --chip='npu' \
        --platform='linux' \
        --train_step=10000
