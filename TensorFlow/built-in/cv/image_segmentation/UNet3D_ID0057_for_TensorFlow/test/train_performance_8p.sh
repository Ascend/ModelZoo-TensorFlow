#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`
#集合通信参数,不需要修改
#source /usr/local/Ascend/CANN-1.81/bin/setenv.bash
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export RANK_SIZES=8
export JOB_ID=10087
#export RANK_TABLE_FILE=$cur_path/../scripts/8p.json
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="UNet3D_ID0057_for_TensorFlow"
batch_size=2
#训练步数
train_steps=500 #640

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
    echo"usage:./train_performance_1p.sh <args>"
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
    elif [[ $para == --one_node_ip* ]];then
        one_node_ip=`echo ${para#*=}`
    fi
done

#8p训练必须参数（本机IP）
one_node_ip=$one_node_ip
#新增适配集群环境变量
export CM_CHIEF_IP=${one_node_ip}   #主节点ip，所有服务器一致
export CM_CHIEF_PORT=29688          #通信端口，所有服务器一致
export CM_CHIEF_DEVICE=0            #配置为0，配置主卡，类似于主节点，所有服务器一致
export CM_WORKER_SIZE=8             #卡数，单机为8，所有服务器一致
export CM_WORKER_IP=${one_node_ip}  #当前服务器ip，不同环境ip不同

#data_path='../'
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cd $cur_path/../
sed -i 's/RANK_SIZE/RANK_SIZES/g' model/model_fn.py pbinference/unet3d_pb_inference.sh main_npu.py dataset/data_loader.py runtime/hooks.py runtime/setup.py
sed -i 's/RANK_ID/RANK_IDS/g' pbinference/unet3d_pb_inference.sh main_npu.py dataset/data_loader.py runtime/hooks.py runtime/setup.py

#训练开始时间，不需要修改
start_time=$(date +%s)
bind_core=1
exec_mode='train'
#进入训练脚本目录，需要模型审视修改
for((RANK_IDS=$RANK_ID_START;RANK_IDS<$((RANK_SIZES+RANK_ID_START));RANK_IDS++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_IDS"
    export RANK_IDS=$RANK_IDS
    export ASCEND_DEVICE_ID=`expr ${RANK_IDS} - ${RANK_ID_START}`
    ASCEND_DEVICE_ID=`expr ${RANK_IDS} - ${RANK_ID_START}`
    export DEVICE_ID=${ASCEND_DEVICE_ID}
    echo 'DEVICE_ID: '$ASCEND_DEVICE_ID
    RANK_ID_core=$RANK_IDS

    export DEVICE_ID=$RANK_IDS
	  DEVICE_INDEX=$RANK_IDS
    export DEVICE_INDEX=${DEVICE_INDEX}

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    if [ ${RANK_ID_core} -gt 7 ];then
        RANK_ID_core=$((RANK_ID_core-8))
    fi

    echo 'RANK_ID_core is: '$RANK_ID_core

    # 执行训练脚本，需要模型审视修改
    corenum=`cat /proc/cpuinf |grep 'processor' |wc -l`
    let a=RANK_ID_core*${corenum}/8
    let b=RANK_ID_core+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi

    echo "data_path is : $data_path"
	#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 main_npu.py --data_dir=$data_path \
        --model_dir=$cur_path/output/${ASCEND_DEVICE_ID} \
        --exec_mode=${exec_mode} \
        --npu_loss_scale=1048576 \
        --max_steps=$train_steps \
        --benchmark \
        --fold=0 \
        --batch_size=$batch_size \
        --augment > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
sed -i 's/RANK_SIZES/RANK_SIZE/g' model/model_fn.py pbinference/unet3d_pb_inference.sh main_npu.py dataset/data_loader.py runtime/hooks.py runtime/setup.py
sed -i 's/RANK_IDS/RANK_ID/g' pbinference/unet3d_pb_inference.sh main_npu.py dataset/data_loader.py runtime/hooks.py runtime/setup.py


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep throughput_train $cur_path/output/0/train_0.log|awk -F 'throughput_train' '{print $2}'|awk -F ':' '{print $2}'|awk '{print $1}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZES}'p'_'perf'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZES}'*1000/'${FPS}'}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep 'global_step:' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'loss:' '{print $2}'|tr -d ','|awk '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
#ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
ActualLoss=None


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZES}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log