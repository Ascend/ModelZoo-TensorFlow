#!/bin/bash
cur_path=`pwd`
export PYTHONPATH=$cur_path/../models/research:$cur_path/../models/research/slim:$PYTHONPATH

#集合通信
export RANK_SIZE=1
export RANK_TABLE_FILE=$cur_path/../configs/${RANK_SIZE}p_${ASCEND_DEVICE_ID}.json
export JOB_ID=10087
RANK_ID_START=0

#数据集参数
data_path="/data"
use_conda=0

#训练参数，需要根据模型修改
Network="SSD-InceptionV2_ID0510_for_TensorFlow"
num_train_steps=300
batch_size=24
ckpt_path=/checkpoints
pipeline_config=$cur_path/../models/research/configs/ssd_inception_v2_coco_1p.config


#帮助提示，需要根据网络修改
if [[ $1 == --help || $1 == -h ]];then 
	echo "usage: ./train_performance_1p.sh <args>"

	echo ""
	echo "parameter explain:
	--num_train_steps           training steps
	--data_path                 source data of training
	--ckpt_path                  pre-checkpoint path
	--pipeline_config           pipeline config path
	--skip_eval  whether to skip eval
    -h/--help             Show help message
	"
	exit 1
fi

#入参设置，需要根据网络修改
for para in $*
do
    if [[ $para == --num_train_steps* ]];then
		num_train_steps=`echo ${para#*=}`
	elif [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	elif [[ $para == --ckpt_path* ]];then
		ckpt_path=`echo ${para#*=}`
	elif [[ $para == --pipeline_config* ]];then
		pipeline_config=`echo ${para#*=}`
    elif [[ $para == --use_conda* ]];then
	    use_conda=`echo ${para#*=}`
	elif [[ $para == --skip_eval* ]];then
	    skip_eval=`echo ${para#*=}`
    fi
done	

if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi



##########################执行训练#########################
start_time=$(date +%s)
cd $cur_path/../models/research
if [  -f ${pipeline_config}.bak ];then
   cp ${pipeline_config}.bak ${pipeline_config}
else
   cp ${pipeline_config} ${pipeline_config}.bak
fi

sed -i "s%/checkpoints%${ckpt_path}%p" ${pipeline_config} 
sed -i "s%/data/coco2017_tfrecords%${data_path}/coco2017_tfrecords%p" ${pipeline_config} 

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
  echo "Device ID: $ASCEND_DEVICE_ID"
  export RANK_ID=$RANK_ID
  if [   -d $cur_path/output/${ASCEND_DEVICE_ID} ];then
     rm -rf $cur_path/output/${ASCEND_DEVICE_ID}
     mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
  else
     mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
  fi

#训练执行脚本，需要根据网络修改
  nohup python3 -u ./object_detection/model_main.py \
       --pipeline_config_path=${pipeline_config} \
       --model_dir=$cur_path/output/${ASCEND_DEVICE_ID} \
       --data_path=${data_path}   \
       --overflow_dump_path=${overflow_dump_path}   \
       --step_dump_path=${step_dump_path}   \
       --alsologtostder \
       --amp \
       --num_train_steps=${num_train_steps}  \
	   --skip_eval=True \
       "${@:1}"  > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##########################业务日志#########################
grep ERROR $HOME/ascend/log/plog/*.log > $cur_path/output/$ASCEND_DEVICE_ID/plog_err.log

###########################性能结果处理#########################
echo "-----------------------Final result------------------------"
#性能FPS计算，需要根据网络修改
#FPS=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $2}'` 
FPS=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'NR==2{print $2}'` 

FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${FPS}'*'${RANK_SIZE}'}'`
echo "Final Performance images/sec : $FPS"

################################E2E训练时长##########################
echo "Final Training Duration sec : $e2e_time"

################################性能看护#############################
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
ActualFPS=${FPS}
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
grep INFO:tensorflow:loss $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'|sed 's/,//g'|sed '/^$/d' >> $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

ActualLoss=`awk 'END {print}' $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`
echo "Network = ${Network}" > $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log

