#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#source npu_set_env.sh
export HCCL_CONNECT_TIMEOUT=300
export RANK_ID=0
export RANK_SIZE=1
#export ASCEND_DEVICE_ID=0
export JOB_ID=NPU20210126

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Densenet_3D_for_TensorFlow"

batch_size=2

#维测参数，precision_mode需要模型审视修改
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_8P_256bs_SGD.sh <args>"
    echo " "
    echo "
    --data_path		             source data of training
	--autotune 					Whether to enable autotune, default is False
    -h/--help		               show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done


cur_path=`pwd`/../
cd $cur_path
if [ -d $cur_path/test/output ];then
    rm -rf $cur_path/test/output/*
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi


#训练开始时间，不需要修改
start_time=$(date +%s)

#执行训练，需要模型审视修改
python3.7 train.py -bs 2 -gpu 0 -mn dense24 -sp dense24_correction -nc True -e 5 -r ${data_path} --autotune $autotune > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1

wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
grep 'patient acc:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log|awk '{print $6}'|sed 's/,//g' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt
#最后一个迭代acc值，不需要修改
train_accuracy=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS='NULL'
#单迭代训练时长，不需要修改
TrainingTime='NULL'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'patient loss:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'|sed 's/,//g' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log