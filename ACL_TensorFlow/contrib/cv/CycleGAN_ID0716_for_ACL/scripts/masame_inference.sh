#!/bin/bash

input_path=${1}
output_path=${2}

cd /home/HwHiAiUser/AscendProjects/tools/msame/out/

./msame --model /home/HwHiAiUser/CycleGAN/om_file/zebra2horse_npu200.om \
        --input ${input_path} \
        --output ${output_path}\
        --outfmt BIN