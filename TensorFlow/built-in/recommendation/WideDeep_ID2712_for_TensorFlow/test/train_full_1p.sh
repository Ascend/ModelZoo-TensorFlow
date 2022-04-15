#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087

RANK_ID_START=0


#基础参数，需要模型审视修改
#Batch Size
batch_size=131072
#网络名称，同目录名称
Network="WideDeep_TF_ID2712_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1

#参数配置
data_path="/npu/traindata/ID2940_CarPeting_TF_WideDeep_TF"
display_step=10


#维持参数，以下不需要修改
over_dump=False


# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --over_dump		         if or not over detection, default is False
    --data_path		         source data of training
    --train_epochs       train epochs
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

##############执行训练##########
if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID/ckpt
fi

#if [ -d $cur_path/../config/1p_$ASCEND_DEVICE.json ];then
#    export RANK_TABLE_FILE=$cur_path/../config/1p_$ASCEND_DEVICE.json
#    export RANK_ID=$ASCEND_DEVICE_ID
#else
#    export RANK_TABLE_FILE=$cur_path/../config/1p.json
#    export RANK_ID=0
#fi
wait

#配置文件备份和修改
cd $cur_path/../
if [  -f configs/config.py.bak ];then
   cp configs/config.py.bak configs/config.py
   rm -f configs/config.py.run
else
   cp configs/config.py configs/config.py.bak
   rm -f configs/config.py.run
fi
sed -i "s%/npu/traindata/ID2940_CarPeting_TF_WideDeep_TF%${data_path}%p" configs/config.py
sed -i "s%./model%$cur_path/output/$ASCEND_DEVICE_ID/ckpt%p" configs/config.py
sed -i "s%display_step = 100%display_step = $display_step%p" configs/config.py
#echo `cat configs/config.py |uniq > configs/config.py; cp -f configs/config.py configs/config.py.run`
cp configs/config.py configs/config.py.run


cd $cur_path/../

start=$(date +%s)
nohup python3 train.py --data_path=$data_path --ckpt_path=$cur_path/output/$ASCEND_DEVICE_ID/ckpt > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#配置文件恢复
mv -f configs/config.py.bak configs/config.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改

FPS=`grep 'fps :'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F' ' '{print $25}' | tail -n 1`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'eval auc' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F' ' '{print $8}' |tail -n 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`
echo "TrainingTime(ms/step) : $TrainingTime"

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
loss=`grep 'loss =' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tr -d '\b\r' | awk -F' ' '{print $9}'|sed 's/,$//'`
echo "${loss}"> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`cat $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt | tail -n 1`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
