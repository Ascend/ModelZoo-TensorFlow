#!/bin/bash
input_a=${1}
input_b=${2}
out=${3}

code_dir=$(cd "$(dirname "$0")"; pwd)
echo "===>>>Python boot file dir: ${code_dir}"

python3 ${code_dir}/flownet2_test.py \
        --input_a=${input_a} \
        --input_b=${input_b} \
        --out=${out} \
        --checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0' \
        --save_flo=False
