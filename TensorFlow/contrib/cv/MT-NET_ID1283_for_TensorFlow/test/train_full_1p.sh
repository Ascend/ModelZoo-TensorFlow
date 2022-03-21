#!/bin/bash

export JOB_ID=10087
export RANK_SIZE=1
export RANK_ID_START=0


cur_path=`pwd`
data_path=""
ckpt_path=""
Network='MT-NET_ID1283_for_TensorFlow'
batch_size=4
iteration=60000
# train_performance_1p.sh perf
# train_full_1p.sh acc
CaseName="${Network}_bs${batch_size}_${RANK_SIZE}p_acc"


if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
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
    elif [[ $para == --iteration* ]];then
        iteration=`echo ${para#*=}`
        echo "${iteration}"
    fi
done


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
        mkdir -p ${cur_path}/output/${ASCEND_DEVICE_ID}/sine
    fi
    nohup python3 -u main.py \
        --datasource=sinusoid --metatrain_iterations=${iteration} \
        --meta_batch_size=${batch_size} --update_lr=0.01 --norm=None --resume=True \
        --update_batch_size=10 --use_T=True --use_M=True --share_M=True \
        --logdir=${cur_path}/output/${ASCEND_DEVICE_ID}/sine \
        --platform=linux --chip=npu ${cur_path}/output/${ASCEND_DEVICE_ID}/ckpt > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
end_time=$(date +%s)
e2e_time=$(( ${end_time} - ${start_time} ))


echo "------------------ Final result ------------------"
BatchSize=${batch_size}
DeviceType=`uname -m`
# getFPS
Time=`grep 'time' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'time ' 'END{print $2}' | awk -F's' '{print $1}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${iteration}'/'${Time}'}'`
ActualFPS=${FPS}
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${iteration}'/'${FPS}'}'`
# getAcc
train_accuracy=`grep 'Iteration' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'postlosses ' '{print $2}' | awk -F',' '{print $1}' | tail -n 10 | awk 'END{print s/10}{s+=$1;}'`
# getLoss
grep 'Iteration' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'postlosses ' '{print $2}' | awk -F',' '{print $1}' > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
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
