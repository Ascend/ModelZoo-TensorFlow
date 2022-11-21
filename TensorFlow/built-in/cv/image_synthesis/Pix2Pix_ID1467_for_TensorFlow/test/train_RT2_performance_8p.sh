#!/bin/bash
cur_path=`pwd`/../

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export JOB_ID=10086
#export ASCEND_DEVICE_ID=0
export RANK_ID=0
RANK_ID_START=0
export RANK_SIZE=8
export RANK_TABLE_FILE="./test/8p.json"

#使能RT2.0
export ENABLE_RUNTIME_V2=1

#mkdir $cur_path/test/output/new_weights
#cp -r /npu/traindata/Pix2Pix_weights/* $cur_path/test/output/new_weights
#基础参数，需要模型审视修改
#Batch Size
batch_size=4
#网络名称，同目录名称
Network="Pix2Pix_ID1467_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=8
#训练epoch，可选
train_epochs=5
#训练step
train_steps=
#学习率
learning_rate=

#参数配置
#data_path="/npu/traindata/Pix2Pix_datas"
#ckpt_path="./weights"
#weights_path="/npu/traindata/Pix2Pix_weights"
data_path="./datasets"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_8p.sh --data_path=./datasets --ckpt_path=./weights"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
   elif [[ $para == --ckpt_path* ]];then
      ckpt_path=`echo ${para#*=}`
   fi
done


##############执行训练##########
cd $cur_path
sed -i "s|./datasets/|${data_path}/|g" prepare.py
sed -i "s|./datasets/|${data_path}/|g" utils.py
#sed -i "s|./weights/|${ckpt_path}/|g" model.py
#sed -i "90s|1200|$train_epochs|g" demo.py
wait

#if [ -d $cur_path/test/output ];then
#   rm -rf $cur_path/test/output/*
#   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
#else
#   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
#fi
#wait

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
    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_ID*${corenum}/8
    let b=RANK_ID+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    #nohup ${bind_core} python3 demo.py --weights_path=${ckpt_path} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
	nohup ${bind_core} python3 demo_rt.py \
        --new_weights_path=$cur_path/test/output/$ASCEND_DEVICE_ID \
        --epochs=${train_epochs} \
        --batch_size=${batch_size} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#参数回改
sed -i "s|${data_path}/|./datasets/|g" prepare.py
sed -i "s|${data_path}/|./datasets/|g" utils.py
sed -i "s|${ckpt_path}/|./weights/|g" model.py
sed -i "90s|$train_epochs|1200|g" demo.py
wait
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#TrainingTime=`grep "time:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $15}' |cut -d ']' -f -1`
#wait
#FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
#echo "Final Performance images/sec : $FPS"

# TrainingTime=`grep "time:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $14}' |cut -d ']' -f -1`
step_sec=(`grep "Epoch*" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END{print $15}' |cut -d 'e' -f 2 |sed 's/: //;s/]//g'`)
wait
# FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
echo=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${RANK_SIZE}'}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${echo}'/'${step_sec}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "acc:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $9}'|cut -d ']' -f 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'RT2'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*8000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "G loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $12}'|cut -d ']' -f 1 >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
