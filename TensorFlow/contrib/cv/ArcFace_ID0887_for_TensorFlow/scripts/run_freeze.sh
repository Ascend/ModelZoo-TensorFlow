#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
code_dir=/home/test_user02/Arcface/prcode/ArcfaceCode/
model_path=/home/test_user02/Arcface/prcode/result/20210820-091812/checkpoints/ckpt-m-12997
result=/home/test_user02/Arcface/prcode/pb_result

current_time=`date "+%Y-%m-%d-%H-%M-%S"`

python3.7 ${code_dir}/freeze_graph.py \
          --code_dir=${code_dir} \
          --model_path=${model_path} \
          --result=${result} 2>&1 | tee ${result}/${current_time}_freeze.log
