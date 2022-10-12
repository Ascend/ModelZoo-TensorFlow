#!/bin/bash


export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0


cur_path=`pwd`
data_path=""
ckpt_path=""
Network="DD-NET_ID1088_for_TensorFlow"
batch_size=433
# train_performance_1p.sh perf
# train_full_1p.sh acc
CaseName="${Network}_bs${batch_size}_${RANK_SIZE}p_perf"


if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
    "
    exit 1
fi
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
        echo "${data_path}"
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
        echo "${ckpt_path}"
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
        echo "${batch_size}"
    fi
done
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


# MOVE SOME DATA
mkdir -p ${cur_path}/../data/JHMDB/
cp -r ${data_path}/* ${cur_path}/../data/JHMDB/
cd $cur_path/../
# START
start_time=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/${ASCEND_DEVICE_ID}/ckpt
    else
        mkdir -p ${cur_path}/output/${ASCEND_DEVICE_ID}/ckpt
    fi
    nohup python3 -u ddnet.py > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
end_time=$(date +%s)
e2e_time=$(( ${end_time} - ${start_time} ))
rm -rf ${cur_path}/../data/JHMDB/
wait


echo "------------------ Final result ------------------"
BatchSize=${batch_size}
DeviceType=`uname -m`
# getFPS
FPS=`grep loss ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F's ' 'END{print $2}' | awk -F'us' '{print $1}'`
ActualFPS=`awk 'BEGIN{printf "%.2f\n",1000*1000/'${FPS}'*'${BatchSize}'}'`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`
# getAcc
train_accuracy=`grep loss ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'val_acc: ' 'END{print $2}'`
# getLoss
grep loss ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'loss: ' '{print $2}' | awk -F' ' '{print $1}' > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print}' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`
echo "Final Performance images/sec : ${FPS}"
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : ${e2e_time}"


echo "Network = ${Network}"                  > ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}"              >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}"             >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}"           >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}"               >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}"             >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}"       >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}"    >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}"           >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}"        >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
