#!/bin/bash
rm -rf ./tensorboard/textcnn/*
# set env

export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/python3.7.5/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/atc/python/site-packages:/usr/local/python3.7.5/lib/python3.7/site-packages/
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin

export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910

#export PRINT_MODEL=0
#export DISABLE_REUSE_MEMORY=1

# for fast training
#unset DUMP_OP
#unset PRINT_MODEL
#unset DUMP_GE_GRAPH
export DISABLE_REUSE_MEMORY=0


export JOB_ID=9998001
export RANK_TABLE_FILE=1p.json
export RANK_ID=0
export RANK_SIZE=1
export DEVICE_ID=4


rm -rf output/*
export PYTHONPATH=$PYTHONPATH:./configs:./tensorflow_extentions:./utils
python3 train.py --model_dir output --configuration "v_1_0_SqNxt_23" --num_examples_per_epoch 10000 --batch_size 256 --num_epochs 20  --training_file_pattern "data/imagenet_TF/train-*" --validation_file_pattern "data/imagenet_TF/validation-*"
