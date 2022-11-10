#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ENABLE_RUNTIME_V2=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="Face-ResNet50_ID1372_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=2
#训练step
train_steps=
#学习率
learning_rate=

#参数配置
data_path=""
#work_dir="$cur_path/estimator_working_dir"
#export_path="$cur_path/outputs/models/000001-first_generation"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh"
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

#sed -i "s|./CACD2000_Crop/|${data_path}/|g" TrainResNet.py
#sed -i "s|./label|${data_path}/label|g" TrainResNet.py

start=$(date +%s)
nohup python3 TrainResNet_rt.py \
				--label_path ${data_path}/label/label_1200.npy \
				--image_name_path ${data_path}/label/name_1200.npy \
                --train_data_path ${data_path}/train_data/1200_data.npy \
				--parentPath ${data_path}/CACD2000_Crop/ \
				--epochs 2 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"


#输出性能FPS，需要模型审视修改
steps_per_s=`grep steps_per_s ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END{print $2}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${steps_per_s}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"


#输出训练精度,需要模型审视修改
train_accuracy="None"
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p_RT2_perf'


##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}


#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Cost $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $7}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt


#最后一个迭代loss值(Read-Only)
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中(Read-Only)
echo "Network = ${Network}"                 > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}"              >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}"             >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}"           >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}"               >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}"             >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}"       >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}"    >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}"           >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}"        >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log