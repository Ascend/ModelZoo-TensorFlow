#!/bin/sh
export PYTHONPATH=$PYTHONPATH:'/home/jiayan/avod/avod_npu_20210604062633'
export PYTHONPATH=$PYTHONPATH:'/home/jiayan/avod/avod_npu_20210604062633/wavedata'
# export ASCEND_SLOG_PRINT_TO_STDOUT=1 
# export ASCEND_GLOBAL_LOG_LEVEL=0 # debug level
# export DUMP_GE_GRAPH=3
# export DUMP_GRAPH_PATH='/home/jiayan/avod/avod_npu_20210604062633/ge_graph_tmp'

export EXPERIMENTAL_DYNAMIC_PARTITION=1


EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd $EXEC_DIR

echo $EXEC_DIR
cd  ${EXEC_DIR}
RESULTS=results/1p


#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${device_id}
rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*
cd ${EXEC_DIR}/${RESULTS}/${device_id}

device_id=${VISIBLE_IDS}
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ./train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${device_id}
echo "DEVICE_INDEX $DEVICE_INDEX" 
echo "RANK_ID $RANK_ID" 
echo "DEVICE_ID $DEVICE_ID" 

cd $EXEC_DIR
pwd
env > ${EXEC_DIR}/${RESULTS}/env_${device_id}.log

#start exec
python3.7 avod/experiments/run_training.py --pipeline_config=avod/configs/pyramid_people_example.config
# python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config --device='0' --data_split='val'

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
fi

