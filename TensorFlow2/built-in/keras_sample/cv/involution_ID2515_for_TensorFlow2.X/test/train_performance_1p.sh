#!/bin/bash
cur_path=`pwd`/../

#设置默认日志级别,不需要修改
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=1024
#网络名称，同目录名称
Network="involution_ID2515_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=10
#训练step
#train_steps=50000
#学习率
#learning_rate=1e-5
#网络类型  convolution or involution
network="convolution"
#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage: ./train_performance_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi

##############执行训练##########
cd $cur_path

#参数修改
#sed -i "s|../data/tfrecord|${data_path}/data/tfrecord|g" ${cur_path}/data/io/read_tfrecord.py
#sed -i "s|PRETRAINED_CKPT = ROOT_PATH + '/|PRETRAINED_CKPT = '${cur_path}/|g" ${cur_path}/libs/configs/cfgs.py

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 involution.py --data_path=$data_path \
        --batch_size=$batch_size \
        --epochs=$train_epochs \
        --Drop_Reminder=False \
        --save_h5=False \
        --network=$network > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
Time=`grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n +2 | awk '{print $3}' |  awk '{sum+=$1} END {print sum/NR}'`
StepTime=`awk 'BEGIN{printf "%.2f\n",'${Time}'/'50'}'`
#FPS计算
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${StepTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "Final Performance sec/step : $StepTime"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "accuracy:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n +2 | awk '{print $15}' | awk ' END {print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n +2 | awk '{print $12}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $NF}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${StepTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
