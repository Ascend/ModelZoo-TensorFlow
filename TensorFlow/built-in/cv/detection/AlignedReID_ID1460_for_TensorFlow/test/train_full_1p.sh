#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=30
#网络名称，同目录名称
Network="AlignedReID_ID1460_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=
#训练step
start_step=1
train_steps=562
#学习率
learning_rate=0.01


if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait


#修改参数
sed -i "s|step = sess.run(global_step)|step = $start_step|g" RES.py
wait

start=$(date +%s)
nohup python3 RES.py --data_dir=${data_path}/data --max_steps $train_steps \
		     --batch_size=$batch_size --learning_rate=$learning_rate --initial_checkpoint2=${data_path}/resnet_v1_101.ckpt > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#修改回改
sed -i "s|step = $start_step|step = sess.run(global_step)|g" RES.py
wait

#输出性能FPS，需要模型审视修改
TrainingTimes=`grep "Time taken"  $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'|tail -1`
TrainingTime=`awk 'BEGIN {printf "%.2f\n", '$TrainingTimes'*1000}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep train_accuracy $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'|cut -c 1-5`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${ActualFPS}'}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Step $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'Train loss' '{print $2}'|tr -d ','|awk '{print $NF }' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log