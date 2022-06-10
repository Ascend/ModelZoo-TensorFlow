#!/bin/bash

cur_path=`pwd`


export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TF_CPP_MIN_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

data_path=""
ckpt_path=""
Network="UDCVO_ID2359_for_TensorFlow"
batch_size=8
n_epoch=1
n_checkpoint=10
# train_performance_1p.sh perf
# train_full_1p.sh acc
CaseName="${Network}_bs${batch_size}_${RANK_SIZE}p_perf"
print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"


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
    elif [[ $para == --max_steps* ]];then
        max_steps=`echo ${para#*=}`
        echo "${max_steps}"
    fi
done
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# MOVE SOME DATA
cd $cur_path/../
cp -r ${data_path}/training ${cur_path}/../
sed -i 's/^data//' training/testcase_image_1500.txt
sed -i "/void_voiced/s!^!${data_path}!" training/testcase_image_1500.txt

sed -i 's/^data//' training/testcase_interp_depth_1500.txt
sed -i "/void_voiced/s!^!${data_path}!" training/testcase_interp_depth_1500.txt

sed -i 's/^data//' training/testcase_validity_map_1500.txt
sed -i "/void_release/s!^!${data_path}!" training/testcase_validity_map_1500.txt

sed -i 's/^data//' training/testcase_intrinsics_1500.txt
sed -i "/void_voiced/s!^!${data_path}!" training/testcase_intrinsics_1500.txt

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
    nohup python3 src/train_voiced.py \
        --train_image_path training/testcase_image_1500.txt \
        --train_interp_depth_path training/testcase_interp_depth_1500.txt \
        --train_validity_map_path training/testcase_validity_map_1500.txt \
        --train_intrinsics_path training/testcase_intrinsics_1500.txt \
        --n_epoch ${n_epoch} \
        --n_batch ${batch_size} \
        --n_height 480 \
        --n_width 640 \
        --n_channel 3 \
        --learning_rates 0.50e-4,0.25e-4,0.12e-4 \
        --learning_bounds 12,16 \
        --occ_threshold 1.5 \
        --occ_ksize 7 \
        --net_type vggnet11 \
        --im_filter_pct 0.75 \
        --sz_filter_pct 0.25 \
        --min_predict_z 0.1 \
        --max_predict_z 8.0 \
        --w_ph 1.00 \
        --w_co 0.20 \
        --w_st 0.80 \
        --w_sm 0.15 \
        --w_sz 1.00 \
        --w_pc 0.10 \
        --pose_norm frobenius \
        --rot_param exponential \
        --n_summary 10 \
        --n_checkpoint ${n_checkpoint} \
        --checkpoint_path ${cur_path}/output/${ASCEND_DEVICE_ID}/ckpt > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
end_time=$(date +%s)
e2e_time=$(( ${end_time} - ${start_time} ))

echo "------------------ Final result ------------------"
BatchSize=${batch_size}
DeviceType=`uname -m`
# getFPS
StepTime=`grep "StepTime: " ${print_log} | tail -n 10 | awk '{print $NF}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${StepTime}'}'`
ActualFPS=${FPS}

# getAcc
 train_accuracy="None"

# getLoss
grep loss ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F'loss: ' 'END{print $2}' | awk -F' ' '{print $1}' > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
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
echo "TrainingTime = ${StepTime}"           >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}"    >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}"           >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}"        >> ${cur_path}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
