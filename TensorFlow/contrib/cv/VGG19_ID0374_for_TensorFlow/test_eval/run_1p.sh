#!/bin/bash

if [ ! -d "result/1p" ]; then
    mkdir -p result/1p
else
    rm -rf result/1p/*
fi

# set env
export HCCL_CONNECT_TIMEOUT=600

# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
export ASCEND_DEVICE_ID=0

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0

currentDir=$(cd "$(dirname "$0")"; pwd)

device_group="0"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: evaluate.sh ${device_phy_id} & " >> result/1p/main.log
    ${currentDir}/evaluate.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all evaluate.sh exit " >> result/1p/main.log