#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/home/HwHiAiUser/pbfile/arcface_tf_310.pb --framework=3 --output=/home/HwHiAiUser/omfile/arcface_tf_310 --soc_version=Ascend310 --input_format=NHWC --output_type=FP32 \
        --input_shape="input:1,112,112,3" \
        --log=info \
        --out_nodes="embd_extractor/BatchNorm_1/Reshape_1:0" \
        --precision_mode=allow_mix_precision