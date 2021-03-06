#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087

# 数据集路径,保持为空,不需要修改
data_path=""
#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert_Clue_Finetune_IFLYTEK_ID1648_for_TensorFlow"
#训练batch_size
train_batch_size=32
#训练ephch
num_train_epochs=1.0
#学习率
learning_rate=2e-5

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

#其他参数
print_loss_step=10
max_seq_length=128
task_name=iflytek

if [[ $1 == --help || $1 == -h ]];then 
	echo "usage: ./train_full_1p.sh <args>"

	echo ""
	echo "parameter explain:
	--task_name           finetune dataset
	--data_path           source data of training
	--train_batch_size    training batch
	--learning_rate       learning_rate
	--num_train_epochs    epochs
	--output_dir          output dir
	-h/--help             Show help message
	"
	exit 1
fi

if [   -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	elif [[ $para == --task_name* ]];then
		task_name=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
		ckpt_path=`echo ${para#*=}`
	elif [[ $para == --max_seq_length* ]];then
		max_seq_length=`echo ${para#*=}`
	elif [[ $para == --print_loss_step* ]];then
		print_loss_step=`echo ${para#*=}`
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
    fi
done	

if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi

if [[ $ckpt_path == */ ]];then
	ckpt_path=${ckpt_path%?}
fi
#############执行训练#########################
start=$(date +%s)


cd $cur_path/../
nohup python3  run_classifier.py \
       --task_name=${task_name} \
       --do_train=True \
       --do_eval=True \
       --data_dir=${data_path}/${task_name} \
       --vocab_file=${ckpt_path}/vocab.txt \
       --bert_config_file=${ckpt_path}/bert_config.json \
       --init_checkpoint=${ckpt_path}/bert_model.ckpt \
       --max_seq_length=${max_seq_length} \
       --train_batch_size=${train_batch_size} \
       --learning_rate=${learning_rate} \
       --num_train_epochs=${num_train_epochs} \
       --output_dir=${cur_path}/output/${output_dir} \
       --precision_mode=${precision_mode} \
       --over_dump=${over_dump} \
       --data_dump_flag=${data_dump_flag} \
       --autotune=${autotune} \
       --profiling=${profiling} \
       --profiling_dump_path=${profiling_dump_path} \
       --print_loss_step=${print_loss_step} > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#############结果处理#########################
#step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| sed -n 20p | awk '{print $2}'`
grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log > step_sec.log
cat step_sec.log |tail -n -20 > step_sec1.log
step_sec=`cat step_sec1.log | grep -a 'INFO:tensorflow:global_step/sec: ' | awk '{sum+=$2} END {print sum/NR}'` 

Accuracy=`grep -a 'eval_accuracy = ' ${cur_path}/output/${output_dir}/dev_results_bert.txt|awk 'END {print $3}'`   
FPS=`awk 'BEGIN{printf "%d\n",'$step_sec' * '$train_batch_size'}'` 

echo "--------Final Result ----------"
echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : $Accuracy"
echo "E2E Training Duration sec : $e2etime"


##冒烟看护字段
BatchSize=${train_batch_size}
DeviceType=`uname -m`


#CaseName=${Network}_${task_name}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量
ActualFPS=$FPS
#单迭代训练时长
#TrainingTime=`expr ${train_batch_size} \* 1000 / ${FPS}`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${train_batch_size}'*1000/'${FPS}'}'`

##获取Loss
grep "INFO:tensorflow:global_step = " $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $6}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
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





















