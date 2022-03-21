#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

dataset=${1}
test_file=${2}
checkpoint=${3}
chip=${4}

code_dir=$(cd "$(dirname "$0")"; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3 ${code_dir}/flownet2_val.py \
        --dataset=${dataset} \
        --result='./log/flownet2' \
        --test_file=${test_file} \
        --batch_size=4 \
        --image_size=436,1024 \
        --checkpoint=${checkpoint} \
        --chip=${chip} \
        --gpu_device='0'
