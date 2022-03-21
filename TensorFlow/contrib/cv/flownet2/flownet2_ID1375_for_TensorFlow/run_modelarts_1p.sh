#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

code_dir=${1}
data_dir=${2}
result_dir=${3}
obs_url=${4}
checkpoint_dir=${5}

code_dir=$(cd "$(dirname "$0")"; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python ${code_dir}/flownet2_train.py \
        --dataset=${data_dir} \
        --result=${result_dir} \
        --obs_dir=${obs_url} \
        --train_file=${data_dir}/train.txt \
        --val_file=${data_dir}/val.txt \
        --batch_size=8 \
        --save_step=10 \
        --image_size=436,1024 \
        --pretrained=${checkpoint_dir}/checkpoints/FlowNet2/flownet-2.ckpt-0 \
        --resume_path=${checkpoint_dir}/ckpt/ \
        --chip='npu' \
        --platform='modelarts' \
        --train_step=50
	
