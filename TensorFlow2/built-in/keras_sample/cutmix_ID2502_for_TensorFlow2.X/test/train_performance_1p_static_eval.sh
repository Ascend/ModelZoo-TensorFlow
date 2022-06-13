#!/bin/bash
cur_path=`pwd`/../

#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="cutmix_ID2502_for_TensorFlow2.X"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=5
#训练step
#train_steps=50000
#学习率
# learning_rate=0.001
# weight_decay=0.0001
#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage: ./train_performance_1p_static_eval.sh"
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
python3 $cur_path/train.py --data_dir=${data_path} \
    --epochs=${train_epochs} \
    --batch_size=${batch_size} \
    --eval_static=True > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#参数回改
#sed -i "s|${data_path}/data/tfrecord|../data/tfrecord|g" ${cur_path}/data/io/read_tfrecord.py
#sed -i "s|PRETRAINED_CKPT = '${cur_path}/|PRETRAINED_CKPT = ROOT_PATH + '/|g" ${cur_path}/libs/configs/cfgs.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep ms/step $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'NR==2' | awk '{print$5}' | tr -cd "[0-9]"`
TrainingTime=`awk 'BEGIN{printf "%.3f\n",'${TrainingTime}'/'1000'}'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
# FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
# train_accuracy=`grep accuracy $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print $11}'`
train_accuracy=`grep s/step $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print$17}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=${TrainingTime}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep loss: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $8}' |grep -v loss > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |tr -d '\b\r'| grep -Eo "loss: [0-9]*\.[0-9]*" | awk -F " " '{print $2}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt  
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
# ActualLoss=`grep s/step $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print$8}'`

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