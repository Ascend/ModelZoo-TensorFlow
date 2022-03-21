#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

code_dir=$(cd "$(dirname "$0")"; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python ${code_dir}/flownet2_train.py \
        --dataset='./data/sintel' \
        --result='./log/flownet2' \
        --train_file='./data/sintel/train.txt' \
        --val_file='./data/sintel/val.txt' \
        --batch_size=4 \
        --save_step=5 \
        --image_size=436,1024 \
        --pretrained='./checkpoints/FlowNet2/flownet-2.ckpt-0' \
        --chip='gpu' \
        --gpu_device='0' \
        --train_step=6 2>&1 | tee ${current_time}_train_gpu.log
