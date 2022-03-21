#!/bin/bash
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/home/HwHiAiUser/CycleGAN/model_file/h2z_for_Ascend310_infer/zebra2horse_npu200.pb \
        --framework=3 \
        --output=./CycleGAN/om_file/zebra2horse_npu200 \
        --soc_version=Ascend310 \
        --input_shape="input_image:256,256,3" \
        --log=info \
        --out_nodes="output_image:0" \
        --output_type=UINT8 \
        --input_format=NHWC \
        --precision_mode=allow_fp32_to_fp16

