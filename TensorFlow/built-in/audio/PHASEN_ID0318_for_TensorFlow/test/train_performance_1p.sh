#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="PHASEN_ID0318_for_TensorFlow"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=8
#训练step
train_steps=10
#学习率
learning_rate=8e-2
RANK_SIZE=1

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path		           source data of training
    --train_steps                  train steps
    -h/--help		             show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
	train_steps=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#进入训练脚本目录，需要模型审视修改
cd $cur_path
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi

sed -i "s|file_name = \"|file_name = \"${data_path}/|g" train.py
sed -i "s|model_name = \"|model_name = \"${data_path}/models/|g" train.py
sed -i "s|steps=10000|steps=${train_steps}|g" train.py

#训练开始时间，不需要修改
start_time=$(date +%s)
ln -s ${data_path} ./nn_se/noisy_datasets_16k
tar zxvf libsndfile-1.0.28.tar.gz
cd libsndfile-1.0.28
chmod +x *
./configure
make -j20
cd ..
source compile.sh
nohup python3 nn_se/_2_train.py ${train_epochs} > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|file_name = \"${data_path}/|file_name = \"|g" train.py
sed -i "s|model_name = \"${data_path}/models/|model_name = \"|g" train.py
sed -i "s|steps=${train_steps}|steps=10000|g" train.py

#输出训练精度,需要模型审视修改
#train_accuracy=`grep train_accuracy $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'|cut -c 1-5`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
#单迭代训练时长
TrainingTime=`grep train\: $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -1 | awk '{print $4}' | sed 's/s/\ /g' | awk '{print $1}'`
#需要修改第三方库，TrainingTime无法获取
ActualFPS=`python3 -c "print(${batch_size}/${TrainingTime})"`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "sum_loss " $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $6}'|cut -d , -f 1 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`grep train\: $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|  tail -1 | awk '{print $6}' | sed 's/,//g'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
