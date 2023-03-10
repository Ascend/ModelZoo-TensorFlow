#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
export JOB_ID=10087
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="NIMA_ID0158_for_TensorFlow"
batch_size=200
#训练epochs
train_epochs=1


# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag		 data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --autotune               whether to enable autotune, default is False
    --data_path		         source data of training
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi



#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
       
#创建DeviceID输出目录，不需要修改
if [ -d $cur_path/output/$ASCEND_DEVICE_ID ];then
	rm -rf $cur_path/output/$ASCEND_DEVICE_ID
	mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
	mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi

if [ -d $cur_path/../weights ];then
	rm -rf $cur_path/../weights
	mkdir -p $cur_path/../weights
else
	mkdir -p $cur_path/../weights
fi

#cp $data_path/AVA.txt  $cur_path/../AVA_dataset

#参数修改
sed -i "s|weights/|$cur_path/../|g" $cur_path/../train_mobilenet.py
sed -i "s|./AVA_dataset/|$data_path/|g" $cur_path/../utils/data_loader.py
wait

#训练开始时间，不需要修改
start_time=$(date +%s)

#执行训练脚本，需要模型审视修改
nohup  python3 train_mobilenet.py > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

	
	
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
tp=`grep 'ms/step' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " 'END{print$5}'`
echo $tp
if [ -z $tp ]; then
  echo "IS NULL"
  train_time=`grep '1250/1250'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 's/step' '{print $1}'|awk '{print $NF}'`
  FPS=`awk 'BEGIN{printf "%.2f\n", '$batch_size'/'$train_time'}'`
else
  echo "NOT NULL"
  train_time=`grep '1250/1250'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'ms/step' '{print $1}'|awk '{print $NF}'`
  FPS=`awk 'BEGIN{printf "%.2f\n", '$batch_size'* 1000/'$train_time'}'`
fi


#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep -a 'accuracy:  ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $2}' | awk -F "%;" '{print $1}'`

#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#参数改回
sed -i "s|$cur_path/../|weights/|g" $cur_path/../train_mobilenet.py
sed -i "s|$data_path/|./AVA_dataset/|g" $cur_path/../utils/data_loader.py
wait

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$train_time

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep ETA  $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tr -d '\b\r'|grep -Eo "loss: [0-9]*\.[0-9]*"|awk '{print $NF}'|cut -c 1-6 > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log