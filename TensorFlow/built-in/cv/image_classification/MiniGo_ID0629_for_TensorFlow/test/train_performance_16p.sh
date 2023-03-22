#!/bin/bash

cur_path=`pwd`/../
rm -f $cur_path/outputs/models/*
rm -f $cur_path/estimator_working_dir/*

#基础参数，需要模型审视修改
#Batch Size
batch_size=128
#网络名称，同目录名称
Network="MiniGo_ID0629_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=16
#训练epoch，可选
train_epochs=
#训练step
train_steps=500
#学习率
learning_rate=
#动态输入模式，不需要修改
dynamic_input=""

#参数配置 npu param
precision_mode="allow_fp32_to_fp16"
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

data_path="$./outputs/data/selfplay"

if [[ $1 == --help || $1 == -h ]];then
   echo "usage: ./train_performance_1p.sh $data_path --work_dir="$cur_path/estimator_working_dir" --export_path="$cur_path/outputs/models/000001-first_generation""
   exit 1
fi

for para in $*
do
   if [[ $para  == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    elif [[ $para == --dynamic_input* ]];then
      dynamic_input=`echo ${para#*=}` 
	elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --devices_num* ]];then
	    devices_num=`echo ${para#*=}`
    elif [[ $para == --servers_num* ]];then
	    servers_num=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
    elif [[ $para == --one_node_ip* ]];then
        one_node_ip=`echo ${para#*=}`
    fi
done

linux_num=$servers_num

if [[ $data_path  == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi
export RANK_SIZES=`awk 'BEGIN{printf "%.0f\n",'${devices_num}'*'${linux_num}'}'`
rank_size=8
#nohup python3 set_ranktable.py --npu_nums=$linux_num --conf_path=$conf_path
#wait
#export RANK_TABLE_FILE=$cur_path/test/rank_table.json
export JOB_ID=10087
export DEVICE_INDEX=0

##############执行训练##########
cd $cur_path

#(Step1)初始化  一定要先运行这一步
python3 bootstrap.py --work_dir=$cur_path/estimator_working_dir --export_path=$cur_path/outputs/models/000000-bootstrap
wait

if [[ $conf_path == "" ]];then
    fix_node_ip=$fix_node_ip
    one_node_ip=$one_node_ip
else
    one_node_ip=`find $conf_path -name "server_*_0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
fi

#新增适配集群环境变量
export CM_CHIEF_IP=${one_node_ip}   #主节点ip，所有服务器一致
export CM_CHIEF_PORT=29688          #通信端口，所有服务器一致
export CM_CHIEF_DEVICE=0            #配置为0，配置主卡，类似于主节点，所有服务器一致
export CM_WORKER_SIZE=16            #卡数，单机为8，多机为8n,所有服务器一致
export CM_WORKER_IP=${fix_node_ip}  #当前服务器ip，不同环境ip不同
start=$(date +%s)

# 8P训练模式
for((RANK_IDS=$((rank_size*server_index));RANK_IDS<$((((server_index+1))*rank_size));RANK_IDS++));
do
    #设置环境变量
    
    export ASCEND_DEVICE_ID=`expr ${RANK_IDS} - $((rank_size*server_index))`
    ASCEND_DEVICE_ID=`expr ${RANK_IDS} - $((rank_size*server_index))`
    export RANK_IDS=$RANK_IDS
    echo "Device ID: $ASCEND_DEVICE_ID"

    if [ -d $cur_path/test/output/$ASCEND_DEVICE_ID ];then
        rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    else
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
    fi
    echo $ASCEND_DEVICE_ID
    #(Step3)训练
    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_IDS*${corenum}/8
    let b=RANK_IDS+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    #${bind_core} python3 train.py --training_data_path=$data_path --steps_to_train=$train_steps --train_batch_size=$batch_size --work_dir=$cur_path/estimator_working_dir --export_path=$cur_path/outputs/models/000001-first_generation > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
	${bind_core} python3 train.py \
		--training_data_path=$data_path \
		--steps_to_train=$train_steps \
		--train_batch_size=$batch_size \
		--work_dir=$cur_path/estimator_working_dir \
		--export_path=$cur_path/outputs/models/000001-first_generation \
		--dynamic_input=${dynamic_input} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"


###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

#获取性能
TrainingTime=`grep "tensorflow:global_step/sec" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
wait
ActualFPS=`awk 'BEGIN{printf "%.2f\n", '${BatchSize}'*'${RankSize}'*'${TrainingTime}'}'`

#从train_*.log中提取Loss到${CaseName}_loss.txt中
grep "] loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $7}' |cut -d , -f 1 >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DynamicInput = ${dynamic_input}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log