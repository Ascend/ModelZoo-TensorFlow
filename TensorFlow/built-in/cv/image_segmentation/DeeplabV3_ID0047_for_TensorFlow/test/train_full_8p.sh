#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../scripts/8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""


#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="DeeplabV3_ID0047_for_TensorFlow"

#训练batch_size
batch_size=8


#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                    if or not over detection, default is False
    --data_dump_flag               data dump flag, default is 0
    --data_dump_step               data dump step, default is 10
    --profiling                    if or not profiling for performance debug, default is False
    --autotune                 whether to enable autotune, default is False
    --data_path                    source data of training
    -h/--help                      show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#autotune时，先开启autotune执行单P训练，不需要修改
if [[ $autotune == True ]]; then
    train_full_1p.sh --autotune=$autotune --data_path=$data_path
    wait
    autotune=False
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    export DEVICE_ID=$ASCEND_DEVICE_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        rm -rf ${cur_path}/output/s8
        mkdir -p ${cur_path}/output/s8/r1/${ASCEND_DEVICE_ID}/
        mkdir -p ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${cur_path}/output/s8/r1/${ASCEND_DEVICE_ID}/
        mkdir -p ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/
    fi


     # 绑核，不需要的绑核的模型删除，需要模型审视修改
    #corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    #let a=RANK_ID*${corenum}/${RANK_SIZE}
    #let b=RANK_ID+1
    #let c=b*${corenum}/${RANK_SIZE}-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    #if [ "x${bind_core}" != x ];then
    #    bind_core="taskset -c $a-$c"
    #fi
    #nohup ${bind_core} python3.7 $cur_path/../train_npu.py \
    cd ${cur_path}/output/s8/r1/${ASCEND_DEVICE_ID}
    python3.7 $cur_path/../train_npu.py \
    --model_variant='resnet_v1_101_beta' \
    --train_split='trainaug' \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=8 \
    --train_crop_size="513,513" \
    --train_batch_size=${batch_size} \
    --training_number_of_steps=15000 \
    --fine_tune_batch_norm=true \
    --tf_initial_checkpoint=${data_path}/pretrained/model.ckpt \
    --log_steps=100 \
    --weight_decay=0.0001 \
    --last_layer_gradient_multiplier=1 \
    --bias_multiplier=2.0 \
    --rank=8 \
    --multi_grid=1 \
    --multi_grid=2 \
    --multi_grid=4 \
    --display_every=500 \
    --iterations_per_loop=500 \
    --aspp_with_separable_conv=False \
    --learning_policy="cosine" \
    --base_learning_rate=0.016 \
    --decay_steps=15000 \
    --dataset_dir=${data_path}/tfrecord \
    > ${cur_path}/output/s8/r1/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    cd -
done
wait

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    export DEVICE_ID=$ASCEND_DEVICE_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

     # 绑核，不需要的绑核的模型删除，需要模型审视修改
    #corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    #let a=RANK_ID*${corenum}/${RANK_SIZE}
    #let b=RANK_ID+1
    #let c=b*${corenum}/${RANK_SIZE}-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    #if [ "x${bind_core}" != x ];then
    #    bind_core="taskset -c $a-$c"
    #fi
    #nohup ${bind_core} python3.7 $cur_path/../train_npu.py \
    cd ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}
    python3.7 $cur_path/../train_npu.py \
    --model_variant='resnet_v1_101_beta' \
    --train_split='train' \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=8 \
    --train_crop_size="513,513" \
    --train_batch_size=${batch_size} \
    --training_number_of_steps=10000 \
    --fine_tune_batch_norm=true \
    --tf_initial_checkpoint=${cur_path}/output/s8/r1/${ASCEND_DEVICE_ID}/resnet_101/model.ckpt-15000 \
    --log_steps=100 \
    --weight_decay=0.00002 \
    --last_layer_gradient_multiplier=1 \
    --bias_multiplier=2.0 \
    --rank=8 \
    --multi_grid=1 \
    --multi_grid=2 \
    --multi_grid=4 \
    --display_every=500 \
    --iterations_per_loop=500 \
    --aspp_with_separable_conv=False \
    --learning_policy="cosine" \
    --base_learning_rate=0.0003 \
    --decay_steps=10000 \
    --dataset_dir=${data_path}/tfrecord \
    > ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    cd -
done
wait
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
     #设置环境变量，不需要修改
     echo "Device ID: $RANK_ID"
     export RANK_ID=$RANK_ID
     export ASCEND_DEVICE_ID=$RANK_ID
     export DEVICE_ID=$ASCEND_DEVICE_ID
     ASCEND_DEVICE_ID=$RANK_ID

     #创建DeviceID输出目录，不需要修改
     if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
         rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
         mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
     else
         mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
     fi
     if [ $ASCEND_DEVICE_ID -eq 7 ];then
         python3.7 $cur_path/../train_npu.py \
         --mode=evaluate \
         --eval_split="val" \
         --model_variant="resnet_v1_101_beta" \
         --iterations_per_loop=1 \
         --atrous_rates=6 \
         --atrous_rates=12 \
         --atrous_rates=18 \
         --output_stride=8 \
         --multi_grid=1 \
         --multi_grid=2 \
         --multi_grid=4 \
         --eval_scales=0.5 \
         --eval_scales=0.75 \
         --eval_scales=1.0 \
         --eval_scales=1.25 \
         --eval_scales=1.5 \
         --eval_scales=1.75 \
         --eval_crop_size="513,513" \
         --aspp_with_separable_conv=False \
         --checkpoint_dir=${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/resnet_101 \
         --dataset_dir=${data_path}/tfrecord \
         --max_number_of_evaluations=1 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1
     fi
done

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep 'FPS:' ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F "=" '{print $1}' | awk 'END {print $6}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'class_20' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F " " '{print $3}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`echo "scale=2;${batch_size} * ${RANK_SIZE} * 1000 / ${FPS}"|bc`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'loss:' ${cur_path}/output/s8/r2/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $7}' | awk -F "loss:" '{print $2}' >> $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log