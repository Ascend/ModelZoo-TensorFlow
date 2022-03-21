#!/bin/bash
#set env
### GPU Platform command for train
export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

code_dir=$(cd "$(dirname "$0")"; cd ..; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 ${code_dir}/train_omniglot.py \
        --dataset=/data/omniglot \
        --result=./checkpoint \
        --chip='gpu' 2>&1 | tee ${current_time}_train_gpu.log