#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=4
export RANK_TABLE_FILE=${cur_path}/../configs/4p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="XLNET_ID0025_for_TensorFlow"
#训练batch_size
batch_size=8
#训练step
train_steps=200
#学习率
learning_rate=5e-5
#其他参数
max_seq_length=128
save_steps=600
warmup_steps=120

#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维测参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
#其他维测参数
random_remove=False
loss_scale_flag=False
loss_scale_value=1

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_4p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_path		         source data of training
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)



#进入训练脚本目录，需要模型审视修改
cd $cur_path/..
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
	echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d $cur_path/output ];then
        rm -rf $cur_path/output/*
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/proc_data/sts-b
  	else
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
		mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/proc_data/sts-b
    fi

    nohup python3 -u ./run_classifier.py \
        --do_train=True \
	    --do_eval=False \
	    --task_name=sts-b \
	    --data_dir=$data_path/STS-B \
	    --output_dir=${cur_path}/output/$ASCEND_DEVICE_ID/proc_data/sts-b \
	    --model_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt \
	    --uncased=False \
	    --spiece_model_file=$ckpt_path/spiece.model\
	    --model_config_path=$ckpt_path/xlnet_config.json \
	    --init_checkpoint=$ckpt_path/xlnet_model.ckpt \
	    --max_seq_length=$max_seq_length \
	    --train_batch_size=$batch_size \
	    --num_hosts=1 \
	    --num_core_per_host=1 \
	    --learning_rate=$learning_rate \
	    --train_steps=$train_steps \
	    --warmup_steps=$warmup_steps \
	    --save_steps=$save_steps \
	    --is_regression=True \
	    --precision_mode=$precision_mode \
        --loss_scale_flag=$loss_scale_flag \
	    --loss_scale_value=$loss_scale_value \
	    --over_dump=$over_dump \
	    --data_dump=$data_dump_flag \
	    --data_dump_step=$data_dump_step \
	    --profiling=$profiling \
	    --random_remove=$random_remove > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy="NA"
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`echo "$TrainingTime" | awk '{printf("%.3f\n",1/$1*1000)}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "INFO:tensorflow:loss" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F"," '{print $1}'|awk '{print $3}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
