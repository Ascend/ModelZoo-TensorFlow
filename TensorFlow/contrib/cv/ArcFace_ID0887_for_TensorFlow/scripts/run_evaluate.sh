#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=/home/test_user02/Arcface/prcode/ArcfaceCode
data_dir=/home/test_user02/Arcface/dataset
result_dir=/home/test_user02/Arcface/prcode/eval_result
model_path=/home/test_user02/Arcface/prcode/result/20210820-091812/checkpoints/ckpt-m-12997

current_time=`date "+%Y-%m-%d-%H-%M-%S"`
python3.7 ${code_dir}/evaluate_npu.py \
        --input_dir=${data_dir} \
        --result=${result_dir} \
        --model_path=${model_path} \
        --code_dir=${code_dir}  2>&1 | tee ${result_dir}/${current_time}_train_npu.log
 