
#!/bin/bash

export RANK_SIZE=1
RANK_SIZE=1
cur_path=`pwd`/../
cd $cur_path
filters=32
dataset='cifar10.5@40-1'
train_dir='./experiments/fixmatch'
less_step=1024
Network="FIXMATCH_ID0843_for_TensorFlow"
ASCEND_DEVICE_ID=0

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:bash ./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --filters                set filters number, default is 32
    --dataset                dataset name and some dataset setting, default is cifar10.5@40-1
    --train_dir		         model's output place, default is ./experiments/fixmatch
    -h/--help		         show help message
    "
    exit 1
fi
for para in $*
do
    if [[ $para == --filters* ]];then
        filters=`echo ${para#*=}`
    elif [[ $para == --dataset* ]];then
        dataset=`echo ${para#*=}`
    elif [[ $para == --train_dir* ]];then
        train_dir=`echo ${para#*=}`
    fi
done

echo "Device ID: $ASCEND_DEVICE_ID"
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi

start=$(date +%s)
python fixmatch.py \
        --filters=${filters} \
        --dataset=${dataset} \
        --train_dir=${train_dir} > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
wait
end=$(date +%s)

e2e_time=$(( $end - $start ))
echo "Final Training Duration sec : $e2e_time"


#输出训练精度,需要模型审视修改
train_accuracy=`grep "accuracy train/valid/test" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


CaseName=${Network}_bs64_${RANK_SIZE}'p'_'perf'
DeviceType=`uname -m`
BatchSize=64
#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log


