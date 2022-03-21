#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

dname=$(dirname "$PWD")

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing evaluate fail" >> ${currentDir}/evaluate_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/1p/${device_id}
rm -rf ${currentDir}/result/1p/${device_id}/*
cd ${currentDir}/result/1p/${device_id}

#start exec
nohup python3.7 ${dname}/train.py --rank_size=1 \
	  --mode='evaluate' \
    --max_train_steps=100 \
    --iterations_per_loop=10 \
    --data_dir=/home/test_user04/ascend/Imagenet2012/ILSVRC2012 \
    --eval_dir=
    --display_every=10 \
    --log_dir=./model_1p \
    --log_name=vgg19_1p.log > ${currentDir}/result/1p/evaluate_${device_id}.log 2>&1 &


if [ $? -eq 0 ] ;
then
    echo "turing evaluate success" >> ${currentDir}/result/1p/evaluate_${device_id}.log
else
    echo "turing evaluate fail" >> ${currentDir}/result/1p/evaluate_${device_id}.log
fi
