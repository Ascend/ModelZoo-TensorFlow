#!/bin/bash

echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p.sh "
export JOB_ID=10086
#export ASCEND_DEVICE_ID=2
export RANK_SIZE=1
#####################
# 【必填】对标性能和精度指标，来源于论文或者GPU复现
benchmark_fps=""
benchmark_accu=""
#####################
# 训练标准化参数列表，请通过参数进行相关路径传递
cur_path=`pwd`
# 1、数据集路径地址，若不涉及则跳过
data_path=''
# 2、验证数据集路径地址，若不涉及则跳过
eval_data_path=''
# 3、预加载checkpoint地址，若不涉及则跳过
ckpt_path=''
# 4、需要加载的其他文件地址，若不涉及则跳过
npu_other_path=''
# 5、训练输出的地址，若不涉及则跳过
output_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`

    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done
#####################
# 训练执行拉起命令，打屏信息输出到train_output.log文件
cd $cur_path/pretrain
python3.7 run_pretraining.py \
  --bert_config_file=$cur_path/Data/Roberta-large/config.json \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_warmup_steps=100 \
  --num_train_steps=100000 \
  --optimizer_type=adam \
  --manual_fp16=True \
  --use_fp16_cls=True \
  --input_files_dir=$data_path \
  --eval_files_dir=$eval_data_path \
  --npu_bert_debug=False \
  --npu_bert_use_tdt=True \
  --do_train=True \
  --num_accumulation_steps=1 \
  --npu_bert_job_start_file= \
  --iterations_per_loop=100 \
  --save_checkpoints_steps=10000 \
  --npu_bert_clip_by_global_norm=False \
  --distributed=False \
  --npu_bert_loss_scale=0 \
  --output_dir=$cur_path/model \
  --out_log_dir=$cur_path/model/loss
#####################
# 【选填】基于当前输出的train_output.log计算出NPU性能和精度值
#npu_fps="请通过train_output.log输出信息，计算出fps性能值"
#npu_accu="请通过train_output.log输出信息，计算出accu性能 值"

echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p.sh "