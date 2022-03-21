#!/bin/bash
cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=10
#网络名称，同目录名称
Network="KinD_ID1132_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=1
export RANK_SIZE=1
#学习率
learning_rate=0.0001

#参数配置
data_path="./LOLdataset"
train_epoch=5
eval_epoch=2

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=./LOLdataset --train_epoch=5 --eval_epoch=2"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
   if [[ $para == --train_epoch* ]];then
      data_path=`echo ${para#*=}`
   fi
   if [[ $para == --eval_epoch* ]];then
      data_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi
if [[ $train_epoch  == "" ]];then
   echo "[Error] para \"train_epoch \" must be config"
   exit 1
fi
if [[ $eval_epoch  == "" ]];then
   echo "[Error] para \"eval_epoch \" must be config"
   exit 1
fi

##############执行训练##########
wait
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

####改写参数####
sed -i "s|glob('./LOLdataset/|glob('${data_path}/|g" decomposition_net_train.py
sed -i "s|epoch = 2000|epoch = ${train_epoch}|g" decomposition_net_train.py
sed -i "s|eval_every_epoch = 200|eval_every_epoch = ${eval_epoch}|g" decomposition_net_train.py

start=$(date +%s)
nohup python3 decomposition_net_train.py > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

####参数回改####
sed -i "s|glob('${data_path}/|glob('./LOLdataset/|g" decomposition_net_train.py
sed -i "s|epoch = ${train_epoch}|epoch = 2000|g" decomposition_net_train.py
sed -i "s|eval_every_epoch = ${eval_epoch}|eval_every_epoch = 200|g" decomposition_net_train.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#输出性能FPS，需要模型审视修改
LastTime=`grep "time:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ':' '{print $3}'|awk '{print $1}'|awk -F ',' '{print $1}'|tail -2|head -n 1`
EndTime=`grep "time:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ':' '{print $3}'|awk '{print $1}'|awk -F ',' 'END {print $1}'`
TrainingTime=`awk 'BEGIN{printf "%.5f\n",'$EndTime'-'$LastTime'}'`
wait
ActualFPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "accuracy:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $8}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"

##获取性能数据，不需要修改
#吞吐量

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ':' '{print $4}'|awk '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log