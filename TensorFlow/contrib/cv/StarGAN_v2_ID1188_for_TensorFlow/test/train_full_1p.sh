#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087

RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="StarGAN_v2_ID1188_for_TensorFlow"
# 训练的batch_size
batch_size=4
# 控制训练时长的参数，视各模型修改---少量epoch
epochs=151000   #从150000续训 200000

# case名称 少量epoch-train_performance_1p.sh传入perf，全量-train_full_1p.sh传入acc
# file_name as your file name


#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，需要修改 file_name as your file name  train_performance_1p or train_full_1p
if [[ $1 == --help || $1 == -h ]];then
    echo "usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    -h/--help		         show help message
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
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
cd $cur_path/../

# 该网络训练脚本需要的文件夹定义 需要修改

#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
   #设置环境变量，不需要修改
   echo "Device ID: $ASCEND_DEVICE_ID"
   export RANK_ID=$RANK_ID
   
   #创建DeviceID输出目录，不需要修改
   if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
      rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
      mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
   else
      mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
   fi
   
   #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
   cp -r ${data_path}/dataset/ckpt/* ./checkpoint/StarGAN_v2_afhq-raw_gan_1adv_1sty_1ds_1cyc/  #续训准备150000
   floder="./checkpoint/StarGAN_v2_afhq-raw_gan_1adv_1sty_1ds_1cyc/"
   files=$(ls $floder)
   for f in ${files}
   do
       echo "${f}"
   done
   echo "-------------------start train-------------------------"
   python3 main.py \
      --phase train \
      --data_path ${data_path}/dataset \
      --dataset afhq-raw \
      --batch_size ${batch_size} \
      --iteration ${epochs} \
      --save_freq 500 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
   wait
   echo "-------------------start test----------------------"
   python3 main.py \
      --phase test \
      --data_path ${data_path}/dataset \
      --dataset afhq-raw > ${cur_path}/output/${ASCEND_DEVICE_ID}/test1_${ASCEND_DEVICE_ID}.log 2>&1 &
   wait
   echo "-------------------start calculate fid ----------------------"
   python3 TTUR/fid.py ./results/StarGAN_v2_afhq-raw_gan_1adv_1sty_1ds_1cyc/1  ${data_path}/dataset/afhq-raw/train/dog/ \
      --inception=${data_path}/dataset/inception-2015-12-05 > ${cur_path}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log 2>&1 &

done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep 'time' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk END'{print $4}'`
FPS=`grep 'fps' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk END'{print $10}'`
#FPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'${single_fps}'}'`
#打印，不需要修改
echo "Final Performance TrainingTime : $TrainingTime"
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
# 如果精度存在 检索出精度
train_accuracy=`grep 'FID:' $cur_path/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk '{print $2}'`
# 如果精度不存在 输出None
#train_accuracy="None"
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'd_loss' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $6}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log