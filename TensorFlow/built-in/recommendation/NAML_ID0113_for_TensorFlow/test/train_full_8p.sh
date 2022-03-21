#!/bin/bash

currentDir="$PWD"
parentDir=$(dirname "$PWD")

source ${currentDir}/env.sh
RANK_ID_START=0

export JOB_ID=10018
export RANK_SIZE=8
export RANK_TABLE_FILE=${currentDir}/hccl_${RANK_SIZE}p.json

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
   export RANK_ID=$RANK_ID
   export ASCEND_DEVICE_ID=$((RANK_ID%RANK_SIZE))
   echo "start train running on device ${ASCEND_DEVICE_ID}"

   if [ ! -d ${currentDir}/output/${ASCEND_DEVICE_ID} ]; then
      mkdir -p ${currentDir}/output/${ASCEND_DEVICE_ID}
   fi

   python3.7 ${parentDir}/recommenders-master/examples/00_quick_start/naml_MIND.py \
             --model_path=${currentDir}/output/${ASCEND_DEVICE_ID} \
             --data_path=${currentDir}/data \
             --epochs=1 \
             --max_steps=1000 > ${currentDir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done

