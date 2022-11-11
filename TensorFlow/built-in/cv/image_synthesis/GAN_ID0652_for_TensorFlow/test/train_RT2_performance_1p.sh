#!/bin/bash
cur_path=`pwd`/../

export ENABLE_RUNTIME_V2=1
#参数配置
#base param
batch_size=128
Network="GAN_ID0652_for_TensorFlow"
RankSize=1
train_epochs=3
train_steps=
learning_rate=

#npu param
#precision_mode="allow_fp32_to_fp16"
precision_mode="allow_mix_precision"
loss_scale=False
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False
#data_save_path="/home/data"
#data_path="data"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=data"
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
#sed -i "s|glob('data|glob('${data_path}/|g" imle.py
wait

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 imle_dataset_rt.py \
    --data_path=${data_path} \
    --epochs=${train_epochs} \
    --batch_size=${batch_size} \
    --precision_mode=${precision_mode} \
    --loss_scale=${loss_scale} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
sed -i 's/\x08//g' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
sed -i 's/\x0d/\n/g' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
average_perf=`grep "ms/step" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' |cut -d m -f 1`

echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

#参数回改
#sed -i "s|glob('${data_path}/|glob('data|g" imle.py

#定义基本信息
Network="GAN_ID0652_for_TensorFlow"
RankSize=1
BatchSize=128
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p_RT2_perf'

#获取性能
TrainingTime=`grep "ms/step" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}' |cut -d m -f 1`
wait
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '1000'*'${BatchSize}'/'${TrainingTime}'}'`

#从train_*.log中提取Loss到${CaseName}_loss.txt中
grep "loss: " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $8}' > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt
sed -i '1d' $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`

#关键信息打印到CaseName.log中
echo "Network = ${Network}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = None" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
