#!/bin/bash
export HCCL_CONNECT_TIMEOUT=300
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

upDir=$(dirname "$PWD")
out_dir=${upDir}/test/output
mkdir -p ${upDir}/test/output/0

# user env
export JOB_ID=NPU20210126
export RANK_SIZES=8
#export RANK_TABLE_FILE=${currentDir}/8p.json

data_dir=$1
fold=$2

device_group="0 1 2 3 4 5 6 7"

if [ x"${fold}" = x"all" ] ;
then
    for device_index in ${device_group}
    do
        RANK_IDS=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/train_accuracy_8p.sh ${data_dir} 0 &
    done

    wait
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold${i} train end"

    python3.7 ${upDir}/runtime/parse_results.py --model_dir ${out_dir} --env accuracy >> ${out_dir}/0/train_0.log &
else
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train start"
    for device_index in ${device_group}
    do
        RANK_IDS=${device_index} ASCEND_DEVICE_ID=${device_index} ${currentDir}/train_accuracy_8p.sh ${data_dir} ${fold} &
    done

    wait
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] fold$fold train end"
fi
