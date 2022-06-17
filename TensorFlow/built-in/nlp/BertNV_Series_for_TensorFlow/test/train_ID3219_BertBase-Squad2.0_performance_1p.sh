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
Network="BertBase-Squad2.0_ID3219_for_TensorFlow"
#训练batch_size
train_batch_size=32

#训练ephch
num_train_epochs=1.0
#学习率
learning_rate=5e-6
#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
over_dump_path=${cur_path}/output/overflow_dump
data_dump_flag=False
data_dump_path=${cur_path}/output/data_dump
enable_exception_dump=False
data_dump_step="0"
profiling=False
autotune=False

if [[ $1 == --help || $1 == -h ]];then 
	echo "usage: ./train_full_1p.sh <args>"

	echo ""
	echo "parameter explain:
	--task_name           finetune dataset
	--data_path           source data of training
	--train_batch_size    training batch
	--learning_rate       learning_rate
	--enable_exception_dump enable_exception_dump
	--num_train_epochs    epochs
	--output_dir          output dir
	-h/--help             Show help message
	"
	exit 1
fi

for para in $*
do
    if [[ $para == --task_name* ]];then
		task_name=`echo ${para#*=}`
	elif [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
		elif [[ $para == --squad_version* ]];then
		squad_version=`echo ${para#*=}`
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
    fi
done	

if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi
model_path=${data_path}/uncased_L-12_H-768_A-12

#训练开始时间，不需要修改
start_time=$(date +%s)
#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt${ASCEND_DEVICE_ID}
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt${ASCEND_DEVICE_ID}
    fi

    nohup python3.7 ${cur_path}/../src/run_squad.py \
      --precision_mode=$precision_mode \
      --vocab_file=${model_path}/vocab.txt \
      --bert_config_file=${model_path}/bert_config.json \
      --init_checkpoint=${model_path}/bert_model.ckpt \
      --do_train=True \
      --train_file=${data_path}/dataset/squad_v2.0_train.tf_record \
      --do_predict=False \
      --predict_file=${data_path}/dataset/dev-v2.0.json \
      --eval_script=${data_path}/dataset/evaluate-v2.0.py \
      --train_batch_size=$train_batch_size \
      --learning_rate=$learning_rate \
      --num_train_epochs=$num_train_epochs \
      --save_checkpoints_steps=1000 \
      --npu_bert_loss_scale=0 \
      --num_train_steps=1000 \
      --output_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt${ASCEND_DEVICE_ID} \
      --version_2_with_negative=True \
      --enable_exception_dump=$enable_exception_dump\
      --data_dump_flag=$data_dump_flag \
      --data_dump_step=$data_dump_step\
      --data_dump_path=$data_dump_path\
      --over_dump=$over_dump \
      --over_dump_path=$over_dump_path > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#############结果处理#########################
#输出性能FPS，需要模型审视修改
FPS=`grep "tensorflow:examples/sec" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print $2}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "f1 =" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $3}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"

##获取性能数据
#吞吐量
ActualFPS=$FPS
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${train_batch_size}'*1000/'${FPS}'}'`

##冒烟看护字段
BatchSize=${train_batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "tensorflow:loss =" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F  " " '{print $3}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改'
ActualLoss=(`awk 'END {print $NF}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`)

#关键性息打印到CaseName.log中
echo "Network = ${Network}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${Accuracy}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log





















