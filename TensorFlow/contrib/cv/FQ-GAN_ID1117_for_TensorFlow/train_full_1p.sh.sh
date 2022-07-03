#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export ASCEND_GLOBAL_LOG_LEVEL=3
python3.7 ${1}/main.py \
        --dataset='selfie2anime' \
        --phase='train' \
        --test_train=True \
        --quant=True \
        --epoch=100 \
        ----iteration=10000

       ###--epoch=5 \
        ###--iteration=100
