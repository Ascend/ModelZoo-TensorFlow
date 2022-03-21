#!/bin/sh
EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd $EXEC_DIR

cd ..
EXEC_DIR=$(pwd)
echo $EXEC_DIR
cd  ${EXEC_DIR}
RESULTS=results/1p

#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${device_id}
rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*
cd ${EXEC_DIR}/${RESULTS}/${device_id}

device_id=$1
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${device_id}
echo "device index"
echo $DEVICE_INDEX
echo "rank id"
echo $RANK_ID
echo "device id"
echo $DEVICE_ID


env > ${EXEC_DIR}/${RESULTS}/env_${device_id}.log

#start exec
TRAIN_DIR=${EXEC_DIR}/main
echo $TRAIN_DIR
cd $TRAIN_DIR

#python3 ${EXEC_DIR}/main/train.py --gpu 0 > ./train${device_id}.log 2>&1 #/main/
#python3 ${EXEC_DIR}/main/train.py --gpu 0 > ./train${device_id}.log 2>&1 #/main/

nohup python train.py "training" --data_root ${EXEC_DIR}/Dataset/chess/ --preset synthesizer/presets/chess.json > ./train${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
fi