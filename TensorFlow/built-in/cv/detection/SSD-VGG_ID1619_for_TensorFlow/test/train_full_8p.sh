#!/bin/bash
cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=64
#网络名称，同目录名称
Network="SSD-VGG_ID1619_for_TensorFlow"
#Device数量,8卡
export JOB_ID=10001
export RANK_TABLE_FILE=${cur_path}/config/rank_table_8p.json
export RANK_SIZE=8
RANK_ID_START=0
#训练epoch，可选
train_epochs=200
#训练step
train_steps=
#学习率
learning_rate=

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_8p.sh --data_path=../"
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

if [ ! -d $cur_path/pascal-voc ];then
   ln -s $data_path/pascal-voc pascal-voc
fi

if [ ! -d $cur_path/vgg_graph ];then
   ln -s $data_path/vgg_graph vgg_graph
fi

start=$(date +%s)
##########执行训练##########
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
   echo "Device ID: $RANK_ID"
   export RANK_ID=$RANK_ID
   export ASCEND_DEVICE_ID=$RANK_ID
   export DEVICE_INDEX=$RANK_ID

   cd $cur_path
   if [ -d $cur_path/test/output/$ASCEND_DEVICE_ID ];then
      rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID
      mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
   else
      mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
   fi

   # 绑核，不需要的绑核的模型删除，需要的模型审视修改
   corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
   let a=RANK_ID*${corenum}/${RANK_SIZE}
   let b=RANK_ID+1
   let c=b*${corenum}/${RANK_SIZE}-1
   bind_core="taskset -c $a-$c"

   nohup ${bind_core} python3 train.py \
          --name $cur_path/test/output/$ASCEND_DEVICE_ID/ckpt \
          --num-workers 1 \
          --epochs ${train_epochs} \
          --data-dir $data_path/pascal-voc \
          --vgg-dir $data_path/vgg_graph \
	      "${@:1}"  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

   sleep 1
done
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_per_sec=`grep "perf" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${step_per_sec}'}'`
wait
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "mAP" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $4}'`
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
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep batches |awk '{print $9}'|cut -d ':' -f 2 >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'|cut -d ':' -f 2 >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
