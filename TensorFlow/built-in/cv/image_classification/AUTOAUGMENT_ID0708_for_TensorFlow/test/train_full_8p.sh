#!/bin/bash
cur_path=`pwd`/../

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export JOB_ID=10086
export ASCEND_DEVICE_ID=0
export RANK_ID=0
RANK_ID_START=0
export RANK_SIZE=8
export RANK_TABLE_FILE="./test/8p.json"

#基础参数，需要模型审视修改
#Batch Size
batch_size=128
#网络名称，同目录名称
Network="AUTOAUGMENT_ID0708_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=8
#训练epoch，可选
train_epochs=200
#训练step
train_steps=
#学习率
learning_rate=
#动态输入模式，不需要修改
dynamic_input=""

#参数配置
data_path="/root/.keras/datasets/cifar-10-batches-py.tar.gz"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_8p.sh --data_path=./datasets"
   exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --dynamic_input* ]];then
        dynamic_input=`echo ${para#*=}` 
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi
##############拷贝数据集##########
if [ -d /root/.keras/datasets/cifar-10-batches-py.tar.gz ];then
   rm -rf /root/.keras/datasets/cifar-10-batches-py.tar.gz
   cp ${data_path}/cifar-10-batches-py.tar.gz /root/.keras/datasets/
else
   mkdir -p /root/.keras/datasets
   cp ${data_path}/cifar-10-batches-py.tar.gz /root/.keras/datasets/
fi
wait

##############执行训练##########
start=$(date +%s)
cd $cur_path
for i in 0 1 2 3 4 5 6 7
do
    export RANK_ID=$i
    export ASCEND_DEVICE_ID=$i
    $ASCEND_DEVICE_ID=$i
    echo "Device ID : $ASCEND_DEVICE_ID"
    if [ -d $cur_path/test/output/$ASCEND_DEVICE_ID ];then
       rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    else
       mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    fi
    echo $ASCEND_DEVICE_ID
    nohup python3 train.py --epochs=${train_epochs} --dynamic_input=${dynamic_input} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "\- loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' |cut -d 's' -f -1`
if echo "${TrainingTime}" | grep -q -E 'm$'
then
    TrainingTime=`grep "\- loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' |cut -d 'm' -f -1`
    FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
else
    FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
fi
wait
#FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "\- accuracy" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $11}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "\- loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $8}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

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
echo "DynamicInput = ${dynamic_input}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log