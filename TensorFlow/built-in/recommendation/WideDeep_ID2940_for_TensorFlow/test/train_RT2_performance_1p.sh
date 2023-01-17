#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export GE_USE_STATIC_MEMORY=1

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087


RANK_ID_START=0

#使能RT2.0
export ENABLE_RUNTIME_V2=1

# 数据集路径,保持为空,不需要修改
data_path=""


#基础参数，需要模型审视修改
#Batch Size
batch_size=131072
#网络名称，同目录名称
Network="WideDeep_ID2940_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=1

#迭代下沉循环次数
iteration_per_loop=0

#参数配置
data_path=""

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
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
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

##############执行训练##########
if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID/ckpt
fi

if [ -f $cur_path/../config/1p_$ASCEND_DEVICE_ID.json ];then
    export RANK_TABLE_FILE=$cur_path/../config/1p_$ASCEND_DEVICE_ID.json
    export RANK_ID=0
else
    export RANK_TABLE_FILE=$cur_path/../config/1p_0.json
    export RANK_ID=0
fi

wait

cd $cur_path/../
start=$(date +%s)
python3 -m trainer.task  \
 --iteration_per_loop=$iteration_per_loop \
 --train_data_pattern=$data_path/outbrain/tfrecords/train/part* \
 --eval_data_pattern=$data_path/outbrain/tfrecords/eval/part* \
 --model_dir=$cur_path/output/$ASCEND_DEVICE_ID/ckpt \
 --transformed_metadata_path=$data_path/outbrain/tfrecords \
 --num_epochs=$train_epochs \
 --benchmark \
 --global_batch_size=$batch_size > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改

#Time=`grep "INFO:tensorflow:global_step/sec: "  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F' ' '{print $2}' | tail -n 2 | head -n +1`
#FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${Time}'}'`
FPS=`grep train_throughput $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "train_throughput :" '{print $2}' | sed s/[[:space:]]//g`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出CompileTime
CompileTime=`grep 'step/s' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -n 2| awk '{sum+=$9} END {print sum}'`

#输出训练精度,需要模型审视修改
train_accuracy=null
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'RT2'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
loss=`grep 'INFO:tensorflow:loss' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tr -d '\b\r' | grep -Eo "INFO:tensorflow:loss = [0-9]*\.[0-9]*" | awk -F' = ' '{print $2}'`
echo "${loss}"> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`cat $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt | tail -n 1`

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
echo "CompileTime = ${CompileTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
