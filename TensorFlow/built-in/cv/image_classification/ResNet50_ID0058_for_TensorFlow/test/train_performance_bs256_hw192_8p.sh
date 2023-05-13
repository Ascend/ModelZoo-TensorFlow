#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZES=8
export JOB_ID=99990001
#export RANK_ID=8p
#export SLOG_PRINT_TO_STDOUT=0
#export RANK_TABLE_FILE=${cur_path}/../configs/8p.json
export HCCL_CONNECT_TIMEOUT=600
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ResNet50_ID0058_for_TensorFlow"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=256
#训练step
train_steps=2000
#学习率
learning_rate=

#维测参数，precision_mode需要模型审视修改
precision_mode="must_keep_origin_dtype"
fp32="--fp32"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --autotune               whether to enable autotune, default is False
    --data_path		         source data of training
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
    elif [[ $para == --hf32 ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --fp32 ]];then
        fp32=`echo ${para#*=}`
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
    elif [[ $para == --one_node_ip* ]];then
        one_node_ip=`echo ${para#*=}`
    fi
done

if [[ ${fp32} == "--hf32" ]];then
  export ENABLE_HF32_EXECUTION=1
fi

#8p训练必须参数（本机IP）
one_node_ip=$one_node_ip
#新增适配集群环境变量
export CM_CHIEF_IP=${one_node_ip}   #主节点ip，所有服务器一致
export CM_CHIEF_PORT=29688          #通信端口，所有服务器一致
export CM_CHIEF_DEVICE=0            #配置为0，配置主卡，类似于主节点，所有服务器一致
export CM_WORKER_SIZE=8             #卡数，单机为8，所有服务器一致
export CM_WORKER_IP=${one_node_ip}  #当前服务器ip，不同环境ip不同

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#修改参数
sed -i "50s|PATH_TO_BE_CONFIGURED|${data_path}|g"  $cur_path/../src/configs/res50_256bs_HW192_8p.py
sed -i "107s|PATH_TO_BE_CONFIGURED|${cur_path}/output/0/d\_solution/ckpt0|g"  $cur_path/../src/configs/res50_256bs_HW192_8p.py

cp data_loader.py $cur_path/../src/data_loader/resnet50/
#autotune时，先开启autotune执行单P训练，不需要修改
if [[ $autotune == True ]]; then
    train_full_1p.sh --autotune=$autotune --data_path=$data_path
    wait
    autotune=False
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
sed -i 's/RANK_SIZE/RANK_SIZES/g' ../src/data_loader/resnet50/data_loader.py
#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
for((RANK_IDS=$RANK_ID_START;RANK_IDS<$((RANK_SIZES+RANK_ID_START));RANK_IDS++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_IDS"
    export RANK_IDS=$RANK_IDS
    export DEVICE_INDEX=$RANK_IDS
    export ASCEND_DEVICE_ID=$RANK_IDS
    ASCEND_DEVICE_ID=$RANK_IDS
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi
    
     # 绑核，不需要的绑核的模型删除，需要模型审视修改
    corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    let a=RANK_IDS*${corenum}/${RANK_SIZES}
    let b=RANK_IDS+1
    let c=b*${corenum}/${RANK_SIZES}-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    nohup ${bind_core} python3.7 ${cur_path}/../src/mains/res50.py --config_file=res50_256bs_HW192_8p \
    --max_train_steps=${train_steps} \
    --iterations_per_loop=100 \
    --debug=True \
    --eval=False \
    --precision_mode ${precision_mode} \
    --model_dir=${cur_path}/output/${ASCEND_DEVICE_ID}/d_solution/ckpt${ASCEND_DEVICE_ID} >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
sed -i 's/RANK_SIZES/RANK_SIZE/g' src/data_loader/resnet50/data_loader.py
#参数改回
sed -i "50s|${data_path}|PATH_TO_BE_CONFIGURED|g"  $cur_path/../src/configs/res50_256bs_HW192_8p.py
sed -i "107s|${cur_path}/output/0/d\_solution/ckpt0|PATH_TO_BE_CONFIGURED|g"  $cur_path/../src/configs/res50_256bs_HW192_8p.py

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`cat ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "FPS:" | awk -F "FPS:" '{print $2}' | awk -F "  loss:" '{print $1}' | tail -n +2 | awk '{sum+=$1} END {print sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep -A 1 top1 $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $3}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZES}'p_hw192'_'fp32'_'perf'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZES}'p_hw192'_'hf32'_'perf'
else
  CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZES}'p_hw192'_'perf'
fi


##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${RANK_SIZES}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "FPS:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "loss: " '{print $2}' | awk -F "total" '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZES}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log