#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#ENV


#集合通信参数,不需要修改
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../config/8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="EfficientNet-B4_ID0162_for_TensorFlow"
#训练epoch
train_epochs=350
#训练batch_size
batch_size=32
#训练step
train_steps=800
#学习率
learning_rate=0.016

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="1"
profiling=False
autotune=False

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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done

#export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVEL=1
#export DUMP_GRAPH_PATH=${cur_path}
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_ID*${corenum}/8
    let b=RANK_ID+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    nohup ${bind_core} python3 ${cur_path}/../main_npu.py \
         --data_dir=${data_path} \
         --mode=train \
         --model_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt \
         --train_batch_size=32 \
         --eval_batch_size=32 \
         --iterations_per_loop=100 \
         --train_steps=800 \
         --transpose_input=False \
         --base_learning_rate=0.128 \
         --model_name=efficientnet-b4  > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
         
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $2}'`

#FPS=`grep "Final Performance images/sec" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $5}'`
#FPS=`expr ${RANK_SIZE} \* ${FPS}`
FPS=`awk -v x="$batch_size" -v y="$step_sec" 'BEGIN{printf "%.2f\n",x*y}'`
FPS=`awk -v x="$RANK_SIZE" -v y="$FPS" 'BEGIN{printf "%.2f\n",x*y}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy="null"
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
#TrainingTime=`expr ${batch_size} \* 1000 / ${FPS}`
TrainingTime=`awk -v x="1" -v y="$step_sec" 'BEGIN{printf "%.2f\n",x/y}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep "Iteration:" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $5}'|sed 's/.$//' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep "logger" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $8}'|sed 's/.$//' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk '{print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|tail -n 3|head -n 1|cut -d ":" -f 2`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log