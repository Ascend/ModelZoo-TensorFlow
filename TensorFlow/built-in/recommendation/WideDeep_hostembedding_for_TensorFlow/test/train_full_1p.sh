#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=1
export RANK_TABLE_FILE=${cur_path}/../widedeep_host1p_smoke/hccl.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

export Batch_Size=16000

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL_ETP_ETP=3
#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="WideDeep_hostembedding_for_TensorFlow"

max_steps=15000


#维持参数，不需要修改
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
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --autotune                 whether to enable autotune, default is False
    --data_path		           source data of training
    -h/--help		           show help message
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
		#开autotune特有环境变量
		autotune=True
		export autotune=True
		export REPEAT_TUNE=True
		export ASCEND_DEVICE_ID=0
		export ENABLE_TUNE_BANK=True
		export TE_PARALLEL_COMPILER=32
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi



echo "device_id:${ASCEND_DEVICE_ID}"
echo $data_path

export RANK_ID=0

cat /etc/hccn.conf > $cur_path/address.txt
device_ip=`grep address_${ASCEND_DEVICE_ID} $cur_path/address.txt|tr -d address_${ASCEND_DEVICE_ID}=`
ifconfig > $cur_path/ip.txt
server_id=`grep 'inet 10' $cur_path/ip.txt|awk 'END {print $2}'`



sed -i 's/"device_id":"0"/"device_id":"'${ASCEND_DEVICE_ID}'"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json
sed -i 's/"device_ip":"192.168.100.101"/"device_ip":"'${device_ip}'"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json
sed -i 's/"server_id":"10.155.111.118"/"server_id":"'${server_id}'"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/..


    

#创建DeviceID输出目录，不需要修改
if [ -d $cur_path/output/$ASCEND_DEVICE_ID ];then
    rm -rf $cur_path/output/$ASCEND_DEVICE_ID
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi

    #执行训练脚本，需要模型审视修改
	

python3.7  widedeep_host1p_smoke/host_widedeep_1p_prec.py --data_dir=$data_path --max_steps=15000 --model_dir=${currentDir}/ckpt |& tee $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#sed -i 's/"device_id":"'${ASCEND_DEVICE_ID}'"/"device_id":"0"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json
#sed -i 's/"device_ip":"'${device_ip}'"/"device_ip":"192.168.100.101"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json
#sed -i 's/"server_id":"'${server_id}'"/"server_id":"10.155.111.118"/' ${cur_path}/../widedeep_host1p_smoke/hccl.json


echo "------------------ Final result ------------------"
#单step时长，需要从train_$ASCEND_DEVICE_ID.log里，通过关键字获取。需要模型审视修改
#step_sec=`grep TimeHistory  $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $6}'`
#echo "Final Performance ms/step : $step_sec"
#计算训练时长，需要模型审视修改
#step_sec=`echo ${step_sec%.*}`
#e2e_sec=`expr ${train_epochs} \* 1281167 / ${step_sec} `
#echo "Final Training Duration sec : $e2e_sec"
#训练精度，需要从train_$ASCEND_DEVICE_ID.log里，通过关键字获取。需要模型审视修改
grep Test_auc $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $6}'|sed '/^$/d' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt
train_accuracy=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_acc.txt`
#echo "Final train_accuracy is ${train_accuracy}"
#E2E训练端到端时长，直接计算，不需要修改
echo "E2E training Duration sec: $e2e_time"

#训练用例信息，不需要修改
#BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=`grep fps $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep 15000|awk 'END {print $8}'`
#单迭代训练时长
TrainingTime=${e2e_time}

##获取Loss，通过train_*.log中关键字，需要根据模型审视
grep log_loss  $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $4}'|sed 's/;//g'|sed '/^$/d' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${Batch_Size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime= ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
