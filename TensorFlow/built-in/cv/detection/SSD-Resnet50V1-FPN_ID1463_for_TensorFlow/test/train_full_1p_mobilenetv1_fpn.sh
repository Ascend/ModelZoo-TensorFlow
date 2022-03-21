#!bin/bash
cur_path=`pwd`
#临时补丁，需要根据网络修改
#cp $ASCEND_OPP_PATH/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json $cur_path/aic-ascend910-ops-info.json.bak -f
#python3 ops_info_patch.py

#环境设置，需要根据网络修改
export PYTHONPATH=$cur_path/../models/research:$cur_path/../models/research/slim:$PYTHONPATH

#集合通信
export RANK_SIZE=1
export RANK_TABLE_FILE=$cur_path/../configs/${RANK_SIZE}p_${ASCEND_DEVICE_ID}.json
export JOB_ID=10087
RANK_ID_START=0

#数据集参数
data_path="/data"


#训练参数，需要根据模型修改
Network="SSD-MobilenetV1-FPN_ID1459_for_TensorFlow"
num_train_steps=100000
batch_size=16
ckpt_path=/checkpoints
pipeline_config=$cur_path/../models/research/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_1gpus.config

#维测参数
overflow_dump=False
overflow_dump_path=$cur_path/output/overflow_dump
step_dump=False
step_dump_path=$cur_path/output/step_dump
check_loss_scale=Flase

#帮助提示，需要根据网络修改
if [[ $1 == --help || $1 == -h ]];then 
    echo "usage: ./train_full_1p.sh <args>"

    echo ""
    echo "parameter explain:
    --num_train_steps           training steps
    --data_path                 source data of training
    --ckpt_path                  pre-checkpoint path
    --pipeline_config           pipeline config path
    --overflow_dump        overflow detection，default is False
    --overflow_dump_path   overflow dump path
    --check_loss_scale     check whether loss scale is valid, default is False
    --step_dump            Dump step data, default is False, can only set when overflow_dump is False
    --step_dump_path      step_dump_path
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
    elif [[ $para == --overflow_dump* ]];then
        overflow_dump=`echo ${para#*=}`
        if [  -d ${overflow_dump_path}  ];then
            echo "overflow dump path: ${overflow_dump_path}"
        else
            mkdir -p ${overflow_dump_path}
        fi
    elif [[ $para == --check_loss_scale* ]];then
        check_loss_scale=`echo ${para#*=}`
    elif [[ $para == --step_dump* ]];then
        step_dump=`echo ${para#*=}`
        if [  -d ${step_dump_path}  ];then
            echo "step dump path: ${step_dump_path}"
        else
            mkdir -p ${step_dump_path}
        fi
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

#sed -i "s%/checkpoints%${ckpt_path}%p" ${pipeline_config} 
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
       --model_dir=$cur_path/output/${ASCEND_DEVICE_ID}/npu_ckpt_mobilenetv1_fpn_${RANK_SIZE}p \
       --data_path=${data_path}   \
       --overflow_dump_path=${overflow_dump_path}   \
       --step_dump_path=${step_dump_path}   \
       --alsologtostder \
       --amp \
       --num_train_steps=${num_train_steps}  \
       "${@:1}"  > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

################################性能结果处理#########################
echo "-----------------------Final result------------------------"
#性能FPS计算，需要根据网络修改
FPS=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $2}'|tail -2|head -1` 

FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${FPS}'*'${RANK_SIZE}'}'`
echo "Final Performance images/sec : $FPS"
################################精度结果处理#########################
#精度计算，需要根据网络修改
train_accuracy=`grep Precision $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep Average |awk 'NR==1 {print $13}'`

#echo 'Final Training Accuracy mAP: $train_accuracy'
################################E2E训练时长##########################
echo "Final Training Duration sec : $e2e_time"

################################性能看护#############################
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
ActualFPS=${FPS}
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
grep INFO:tensorflow:loss $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'|sed 's/,//g'|sed '/^$/d'|head -n 1001 >> $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

ActualLoss=`awk 'END {print}' $cur_path/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`
echo "Network = ${Network}" > $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/${ASCEND_DEVICE_ID}/${CaseName}.log