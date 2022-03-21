#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`/../

data_path=
#集合通信参数,不
export RANK_SIZE=1
export JOB_ID=10087
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="SR_EA_ID2128_for_TensorFlow"
#训练epoch
#训练batch_size
batch_size=50
for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
   
   if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
	  source set_conda.sh
	  source activate $conda_name
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改

if [ -d ${cur_path}/test/output ];then
      rm -rf ${cur_path}/test/output/*
      mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID
else
      mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID
fi
cd $cur_path
sed -i "s|/cache/datasets|${data_path}/|g" configs/sr_ea.yml
nohup vega ${cur_path}/configs/sr_ea.yml -b t -d NPU > \
    ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
sed -i "s|${data_path}/|/cache/datasets|g" configs/sr_ea.yml
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#FPS=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'` 
grep "tensorflow:global_step/sec:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F 'global_step/sec:' '{print $2}' > FPS.log
FPS=`cat FPS.log | awk '{sum+=$1} END {print sum/NR}'`

FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "'PSNR'" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "PSNR': |}" 'END {print $2}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$npu_id.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'INFO loss' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $6}'|awk -F , '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
#echo "TrainAccuracy = ${train_accuracy}">> $cur_path/output/$npu_id/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

# 退出anaconda环境
if [ -n "$conda_name" ];then
   conda deactivate
fi
