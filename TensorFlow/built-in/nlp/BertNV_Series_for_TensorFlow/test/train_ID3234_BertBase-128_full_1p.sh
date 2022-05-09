#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
export GE_USE_STATIC_MEMORY=1
export HCCL_CONNECT_TIMEOUT=600
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BertBase-128_ID3234_for_Tensorflow"
#训练batch_size
train_batch_size=32
#训练ephch
num_train_epochs=3.0
#学习率
learning_rate=1e-6
warmup_proportion=0.1
precision="fp32"
#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
optimizer_type="adam"
#维持参数，以下不需要修改
over_dump=False
over_dump_path=${cur_path}/output/${ASCEND_DEVICE_ID}/overflow_dump
data_dump_flag=False
data_dump_path=${cur_path}/output/${ASCEND_DEVICE_ID}/data_dump
enable_exception_dump=False
data_dump_step="0"
profiling=False
autotune=False

#其他参数
task_name=MNLI
output_dir=ckpt
type=official
use_xla=false
use_fp16=""
if [ "$precision" = "fp16" ] ; then
    echo "fp16 activated!"
    use_fp16="--amp"
else
    echo "fp32/tf32 activated!"
    use_fp16="--noamp"
fi


if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag="--nouse_xla"
fi

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_path		         source data of training
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --task_name* ]];then
        task_name=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]];then
        bind_core=`echo ${para#*=}` 
        name_bind="_bindcore"
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    elif [[ $para == --train_batch_size* ]];then
        train_batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
        learning_rate=`echo ${para#*=}`
    elif [[ $para == --num_train_epochs* ]];then
        num_train_epochs=`echo ${para#*=}`
    elif [[ $para == --output_dir* ]];then
        output_dir=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --optimizer_type* ]];then
        optimizer_type=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
    elif [[ $para == --data_dump_path* ]];then
        data_dump_path=`echo ${para#*=}`
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --enable_exception_dump* ]];then
        enable_exception_dump=`echo ${para#*=}`
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
    fi
done    

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi
bertmodelpath=$ckpt_path/uncased_L-12_H-768_A-12
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID}

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${data_dump_path}
        mkdir -p ${over_dump_path}
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${data_dump_path}
        mkdir -p ${over_dump_path}
    fi

    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    let a=RANK_ID*12
    let b=RANK_ID+1
    let c=b*12-1


  nohup ${bind_core} python3 ./src/run_classifier.py \
     --task_name=$task_name \
     --do_train=true \
     --do_eval=true \
     --enable_exception_dump=$enable_exception_dump\
     --data_dump_flag=$data_dump_flag \
     --data_dump_step=$data_dump_step\
     --data_dump_path=$data_dump_path\
     --over_dump=$over_dump \
     --over_dump_path=$over_dump_path \
     --precision_mode=$precision_mode \
     --data_dir=${data_path}/Glue/${task_name} \
     --vocab_file=$bertmodelpath/vocab.txt \
     --bert_config_file=$bertmodelpath/bert_config.json \
     --init_checkpoint=$bertmodelpath/bert_model.ckpt \
     --max_seq_length=128 \
     --train_batch_size=$train_batch_size \
     --learning_rate=$learning_rate \
     --num_train_epochs=$num_train_epochs \
     --output_dir=${cur_path}/output/$ASCEND_DEVICE_ID/${output_dir} \
     --horovod=false "$use_fp16" \
     --distributed=False \
     --npu_bert_loss_scale=0 \
     --optimizer_type= $optimizer_type \
     $use_xla_tag --warmup_proportion=$warmup_proportion  > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
FPS=`awk 'BEGIN{printf "%d\n",'$step_sec' * '$train_batch_size'}'` 
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'eval_accuracy' ${cur_path}/${output_dir}/eval_results.txt|awk '{print $3}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${train_batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep ' basic_session_run_hooks.py:260] loss =' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'loss = ' '{print $2}'|awk -F ', ' '{print $1}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "train_accuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log





