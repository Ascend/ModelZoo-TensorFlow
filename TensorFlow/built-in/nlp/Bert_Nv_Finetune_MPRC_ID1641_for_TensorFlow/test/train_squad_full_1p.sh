#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
#export DEVICE_INDEX=0

# 数据集路径,保持为空,不需要修改
data_path=""
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_1p.json
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert_Nv_Finetune_for_Tensorflow"
#训练batch_size
train_batch_size=16
#训练ephch
num_train_epochs=3.0
#学习率
learning_rate=1.05e-6
warmup_proportion=0.1
precision="fp32"
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


#其他参数
task_name=MRPC
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

if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi
bertmodelpath=$ckpt_path/uncased_L-24_H-1024_A-16
#############执行训练#########################
start=$(date +%s)


if [   -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
   mkdir -p ${data_dump_path}
   mkdir -p ${over_dump_path}
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
   mkdir -p ${data_dump_path}
   mkdir -p ${over_dump_path}
fi

if [ -d ${cur_path}/${output_dir} ];then
  rm -rf ${cur_path}/${output_dir}
fi

cd $cur_path/../
nohup python run_classifier.py \
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
  --data_dir=${data_path}/$task_name \
  --vocab_file=$bertmodelpath/vocab.txt \
  --bert_config_file=$bertmodelpath/bert_config.json \
  --init_checkpoint=$bertmodelpath/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=$train_batch_size \
  --learning_rate=$learning_rate \
  --num_train_epochs=$num_train_epochs \
  --output_dir=${cur_path}/${output_dir} \
  --horovod=false "$use_fp16" \
  $use_xla_tag --warmup_proportion=$warmup_proportion  > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#############结果处理#########################
step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
Accuracy=`grep -a 'eval_accuracy' ${cur_path}/${output_dir}/eval_results.txt|awk '{print $3}'`
   
FPS=`awk 'BEGIN{printf "%d\n",'$step_sec' * '$train_batch_size'}'` 

echo "--------Final Result ----------"
echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : $Accuracy"
echo "E2E Training Duration sec : $e2etime"


##冒烟看护字段
BatchSize=${train_batch_size}
DeviceType=`uname -m`

#if [[ $model_path =~ base ]]||[[ $model_path =~ Base ]]||[[ $model_path =~ BASE ]]
#then
#  model=bertbase
#  else
#  model=bertlarge
#fi

CaseName=${Network}_${task_name}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量
ActualFPS=$FPS
#单迭代训练时长
#TrainingTime=`expr ${train_batch_size} \* 1000 / ${FPS}`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${train_batch_size}'*1000/'${FPS}'}'`

##获取Loss
grep "INFO:tensorflow:Saving dict for global step" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $18}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
ActualLoss=`awk 'END {print $1}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键性息打印到CaseName.log中
echo "Network = ${Network}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${Accuracy}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}">>$cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log





















