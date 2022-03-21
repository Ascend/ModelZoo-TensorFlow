#!/bin/bash
platform=${1}
set -e
today="$(date '+%d_%m_%Y_%T')"

if [ $platform = "npu" ]
then
    code_dir=${2}
    checkpoint_dir=${3}
    model_dir=${4}
    python3.7  ${code_dir}/export_graph.py \
                --checkpoint_dir ${checkpoint_dir} \
                --model_dir ${model_dir} \
                --XtoY_model Hazy2GT_${today}.pb \
                --YtoX_model GT2Hazy_${today}.pb \
                --image_size1 256 \
                --image_size2 256
else
    CUDA_VISIBLE_DEVICES="" python3 export_graph.py --checkpoint_dir checkpoints/Hazy2GT \
                        --model_dir models \
                        --XtoY_model Hazy2GT_$today.pb \
                        --YtoX_model GT2Hazy_$today.pb \
                        --image_size1 256 \
                        --image_size2 256
fi
