#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=1
#export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0
RANK_SIZE=1

# 数据集路径,保持为空,不需要修改
data_path=""

anno_converted='/npu/traindata/COCO2017/val2017.txt'
gt_anno_path='/npu/traindata/COCO2017/annotations/instances_val2017.json'

#屏蔽TF2.4升级到TF2.6图差异带来的性能下降
#export NPU_EXECUTE_OP_BY_ACL=false

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL_ETP=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="YOLOv5_ID1719_for_TensorFlow2.X"

# 训练epoch
stage1_epoch=0
stage2_epoch=1

# 训练batchsize
batch_size=8

train_worker_num=8

# TF2.X独有，不需要修改
export NPU_LOOPSIZE=1

# 精度模式
precision_mode='allow_mix_precision'
#维持参数，不需要修改
over_dump=False
over_dump_path=''
data_dump_flag=False
data_dump_path=''
data_dump_step="1"
profiling=False
autotune=False
perf=20

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		           show help message
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
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be specified"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
bind_core=1
#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/$ASCEND_DEVICE_ID ];then
        rm -rf ${cur_path}/output/$ASCEND_DEVICE_ID
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi
    cd ${cur_path}/output/$ASCEND_DEVICE_ID/
    #执行训练脚本，需要模型审视修改
    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_ID*${corenum}/8
    let b=RANK_ID+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    #${bind_core} python3 ../../../train.py --weights='' \
    nohup ${bind_core} python3 ../../../train.py --weights='' \
              --perf=$perf \
              --model=yolov5m \
              --rank=${RANK_ID} \
              --rank_size=${RANK_SIZE} \
              --train_worker_num=${train_worker_num} \
              --data_path=${data_path} \
              --anno_converted=${anno_converted} \
              --gt_anno_path=${gt_anno_path} \
              --batch_size=${batch_size} \
              --precision_mode=${precision_mode} \
              --stage1_epoch=${stage1_epoch} \
              --stage2_epoch=${stage2_epoch} \
              --over_dump=${over_dump} \
              --over_dump_path=${over_dump_path} \
              --data_dump_flag=${data_dump_flag} \
              --data_dump_step=${data_dump_step} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
#输出性能FPS。需要模型审视修改
epoch_duration=`grep epoch_duration $cur_path/output/0/train_0.log | awk '{print $2}'`
first_step=`grep duration: $cur_path/output/0/train_0.log |head -1| awk -F "duration:" '{print $2}' |sed s/[[:space:]]//g`
FPS=`awk 'BEGIN{printf "%.2f\n",('$perf'+'$train_worker_num'-2)/('$epoch_duration'-'$first_step')*'$batch_size'*1}'`
echo "Final Performance imgs/sec : $FPS"

#训练精度，需要从train_$ASCEND_DEVICE_ID.log里，通过关键字获取。需要模型审视修改
# li=`cat $cur_path/output/0/train_0.log | wc -l`
# num=$(($li - 1))
# train_accuracy=`sed -n "${num}p" $cur_path/output/0/train_0.log | awk '{print $3}'`
# echo "Final Train Accuracy : ${train_accuracy}"
#E2E训练端到端时长，直接计算，不需要修改
echo "E2E training Duration sec: $e2e_time"

#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${batch_size}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",('$epoch_duration'-'$first_step')/('$perf'+'$train_worker_num'-2)}'`

##获取Loss，通过train_*.log中关键字，需要根据模型审视
grep loss $cur_path/output/0/train_0.log|awk '{print $13}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`grep total_loss: $cur_path/output/0/train_0.log | awk 'END{print $13}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    sed -i "/AttributeError/d" $cur_path/output/${RANK_ID}/train_${RANK_ID}.log
done