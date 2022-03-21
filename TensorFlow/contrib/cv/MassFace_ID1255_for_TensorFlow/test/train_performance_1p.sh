#!/bin/bash


export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0


cur_path=`pwd`
data_path=""
ckpt_path=""
Network="MassFace_ID1255_for_TensorFlow"
batch_size=90
epochs=3
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
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
        echo "${epochs}"
    fi
done
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


cd $cur_path/../
# COPY SOME DATA ETC
cd /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops
if [[ 0 == $? ]];then
    echo `pwd`
    cp /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py.bak
    cd -
    cp -rf ./nn_grad.py /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py
    echo '备份源码并替换源码' > this_log.txt
    diff ./nn_grad.py /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py
else
    echo '路径不对,不能替换源码'
    exit 0
fi

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
    nohup python3 -u train_triplet.py \
        --max_nrof_epochs=1 \
        --epoch_size=3 \
        --data_dir=${data_path}/dataset/casia-112x112 \
        --pretrained_model=${ckpt_path}/pretrain_model/model-20211224-112508.ckpt-60000 \
        --learning_rate_schedule_file=${cur_path}/../lr_coco.txt \
        --models_base_dir=${cur_path}/output/${ASCEND_DEVICE_ID}/models \
        > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    wait

    nohup python3 -u test.py --lfw_dir=${data_path}/dataset/lfw-112x112 \
        --lfw_pairs=${data_path}/dataset/pairs.txt \
        --model=${cur_path}/output/${ASCEND_DEVICE_ID}/models/ >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
end_time=$(date +%s)
e2e_time=$(( ${end_time} - ${start_time} ))

BatchSize=${batch_size}
DeviceType=`uname -m`
# getFPS
Time=`grep 'Total Loss' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'Time' 'END{print $2}' | awk -F' ' '{print $1}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${Time}'}'`
ActualFPS=${FPS}
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`
# getAcc
train_accuracy=`grep Accuracy ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'Accuracy: ' '{print $2}' | awk -F'+' '{print $1}'`
# getLoss
grep 'Triplet Loss' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'Triplet Loss ' '{print $2}' | awk -F' ' '{print $1}' > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
ActualLoss=`awk 'END{print}' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : ${e2e_time}"

#关键信息打印到${CaseName}.log中(Read-Only)
echo "Network = ${Network}"                 > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}"              >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}"             >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}"           >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}"               >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}"             >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}"       >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}"    >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}"           >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}"        >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
cp -rf /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py.bak /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py
echo '恢复替换源码' >>  this_log.txt
diff /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py.bak /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py