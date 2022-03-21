#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path="./datasets"
# ckpt_path=""
#基础参数，需要模型审视修改

#网络名称，同目录名称
Network="Pix2Pix_ID1467_for_TensorFlow"

#训练epoch
train_epochs=1200
#训练batch_size
batch_size=4
#训练step
train_steps=
#学习率
learning_rate=

#TF2.X独有，不需要修改
export NPU_LOOP_SIZE=${train_steps}
#维测参数，precision_mode需要模型审视修改
# precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
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
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    elif [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

################################ 执行训练 ##########################

cd $cur_path
#dataset路径在代码里写死，sed替换
sed -i "s|./datasets/|${data_path}/|g" prepare.py
sed -i "s|./datasets/|${data_path}/|g" utils.py
#sed -i "s|./weights/|${ckpt_path}/|g" model.py

if [ -d $cur_path/test/output ];then
    rm -rf $cur_path/test/output/*
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

if [ -d $cur_path/test/output/new_weights ];then
    rm -rf $cur_path/test/output/new_weights/*
else
    mkdir -p $cur_path/test/output/new_weights
fi
wait

# 训练开始时间，不需要修改
start=$(date +%s)
echo "data_path=${data_path};epochs=${train_epochs};batch_size=${batch_size}"
nohup python3 demo.py \
    --epochs=${train_epochs} \
    --batch_size=${batch_size} > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 & 
wait

#训练结束时间，不需要修改
end=$(date +%s)
e2e_time=$(( $end - $start ))
echo "E2E Training Duration sec : $e2e_time"

#参数回改
sed -i "s|${data_path}/|./datasets/|g" prepare.py
sed -i "s|${data_path}/|./datasets/|g" utils.py
#sed -i "s|${ckpt_path}/|./weights/|g" model.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
# TrainingTime=`grep "time:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $14}' |cut -d ']' -f -1`
step_sec=(`grep "Epoch*" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END{print $15}' |cut -d 'e' -f 2 |sed 's/: //;s/]//g'`)
wait
# FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${step_sec}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
# train_accuracy=`grep "acc:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $9}'|cut -d ']' -f 1`
train_accuracy=(`grep "Epoch*" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $9}'|cut -d% -f1 |sed -r 's/.*(.{3})/\1/'`%)
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
# TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`
TrainingTime=${step_sec}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
# grep "G loss" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $12}'|cut -d ']' -f 1 >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep "Epoch*" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk '{print $12}' |cut -d: -f 4 |sed 's/].*//g' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
# ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
gloss_result=(`grep "Epoch*" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk 'END{print $15}' |cut -d: -f 4 |sed 's/].*//g'`)
ActualLoss=${gloss_result}

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
