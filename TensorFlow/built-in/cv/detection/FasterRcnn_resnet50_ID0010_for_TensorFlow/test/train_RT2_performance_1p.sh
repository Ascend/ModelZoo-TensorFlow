#!/bin/bash

export JOB_ID=10000
export ENABLE_RUNTIME_V2=1

exec_mode='train' # or 'train_and_eval'
eval_after_training=False

backbone='resnet50'
backbone_ckpt_path='/npu/traindata/resnet50_ckpt'
data_path='/npu/traindata/coco_official_2017'

batch_size=2
steps=1000

learning_rate_type='cosine' # or 'step'
learning_rate=0.003
warmup_learning_rate=0.00025
warmup_steps=16000
learning_rate_levels='[0.0003, 0.00003]'
learning_rate_steps='[480000, 640000]'

precision_mode='allow_mix_precision'
loss_scale_flag=0
loss_scale_value=256
overflow_dump=False

########## params from command line ##########

for arg in $* ; do
  if [ ${arg:0:2} == '--' ]; then
    arg=${arg:2}
    pos=`expr index "$arg" =`
    if [ $pos > 0 ]; then
      var_name=${arg:0:$pos-1}
      var_value=${arg:$pos}
      eval $var_name=$var_value
    fi
  fi
done

if [ ! $output_dir ]; then
  output_dir="`pwd`/output/"
fi
echo output_dir=$output_dir

training_file_pattern=${training_file_pattern:-$data_path'/tfrecord/train*'}
validation_file_pattern=${validation_file_pattern:-$data_path'/tfrecord/val*'}
val_json_file=${val_json_file:-$data_path'/annotations/instances_val2017.json'}

########## build params_override ##########

unset params_override
params_override=${params_override}backbone=$backbone,
params_override=${params_override}checkpoint="'$backbone_ckpt_path'",
params_override=${params_override}training_file_pattern="'$training_file_pattern'",
params_override=${params_override}validation_file_pattern="'$validation_file_pattern'",
params_override=${params_override}val_json_file="'$val_json_file'",
params_override=${params_override}train_batch_size=$batch_size,
params_override=${params_override}total_steps=$steps,
params_override=${params_override}learning_rate_type=$learning_rate_type,
params_override=${params_override}init_learning_rate=$learning_rate,
params_override=${params_override}warmup_learning_rate=$warmup_learning_rate,
params_override=${params_override}warmup_steps=$warmup_steps,
params_override=${params_override}learning_rate_levels="'$learning_rate_levels'",
params_override=${params_override}learning_rate_steps="'$learning_rate_steps'",
params_override=${params_override}npu_precision_mode=$precision_mode,
params_override=${params_override}npu_loss_scale_flag=$loss_scale_flag,
params_override=${params_override}npu_loss_scale=$loss_scale_value,
params_override=${params_override}npu_overflow_dump=$overflow_dump,

echo [params_override] "$params_override"

########## prepare environment ##########

export RANK_SIZE=1

if [ ! $RANK_ID_START ]; then
  if [ $ASCEND_DEVICE_ID ]; then
    RANK_ID_START=$ASCEND_DEVICE_ID
  elif [ $DEVICE_ID ]; then
    RANK_ID_START=$DEVICE_ID
  else
    RANK_ID_START=0
  fi
fi
export RANK_ID_START
echo "RANK_ID_START="$RANK_ID_START

BASE_PATH=`cd $(dirname $0); pwd`/../FasterRcnn
echo "BASE_PATH="$BASE_PATH

########## run ##########

start_time=$(date +%s)

pids=
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
  echo
  /usr/local/Ascend/driver/tools/msnpureport -d $RANK_ID -g error

  TMP_PATH=$output_dir/$RANK_ID
  mkdir -p $TMP_PATH
  cd $TMP_PATH

  rm -f configs
  ln -s $BASE_PATH/configs configs

  export RANK_ID
  export DEVICE_ID=$RANK_ID
  export ASCEND_DEVICE_ID=$RANK_ID
  export DEVICE_INDEX=$RANK_ID

  python3 $BASE_PATH/mask_rcnn_main.py --mode=$exec_mode \
                                       --eval_after_training=$eval_after_training \
                                       --model_dir=$TMP_PATH/result \
                                       --num_gpus=$RANK_SIZE \
                                       --params_override="$params_override" \
                                       $@ 2>&1 | tee $TMP_PATH/train_${RANK_ID}.log &

  pids[$RANK_ID-$RANK_ID_START]="$RANK_ID $!"
  cd -
done

sleep 1
echo "########## Waiting for pids: "${pids[*]}

for pid in "${pids[@]}"; do
  pid=($pid)
  RANK_ID=${pid[0]}
  pid=${pid[1]}

  wait $pid
  ret=$?
  echo "******************** train finished ******************** $RANK_ID - $pid - ret : $ret"

  ############################## E2E训练时长 ##############################
  end_time=$(date +%s)
  e2e_time=$(( $end_time - $start_time ))
  echo "Final Training Duration sec : $e2e_time"

  ############################## 业务日志 ##############################
  grep ERROR /root/ascend/log/plog/plog-${pid}_*.log > $output_dir/$RANK_ID/plog_err.log

  log_file=$output_dir/$RANK_ID/train_${RANK_ID}.log

  ############################## 性能结果处理 ##############################
  echo "-------------------- Final result --------------------"
  #性能FPS计算，需要根据网络修改
  FPS=`grep -a 'INFO:tensorflow:global_step/sec: ' $log_file|awk 'END {print $2}'`
  FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${FPS}'}'`
  echo "Final Performance images/sec : $FPS"

  #输出CompileTime
  CompileTime=`grep 'batch_time' $log_file|head -n 2|awk '{sum+=$5} END {print sum}'`

  ############################## 精度结果处理 ##############################
  #精度计算，需要根据网络修改
  train_accuracy=`grep "Average Precision" $log_file | awk 'NR==1 {print $NF}'`
  if [ $train_accuracy ]; then
    echo "Final Training Accuracy mAP: $train_accuracy"
  fi

  ############################## 性能看护 ##############################

  Network=FasterRcnn_resnet50_ID0010_for_TensorFlow

  DeviceType=`uname -m`
  CaseName=${Network}_${backbone}_bs${batch_size}_${RANK_SIZE}'p'_'RT2'_'perf'
  ActualFPS=${FPS}
  TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

  # 提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
  grep "INFO:tensorflow:loss" $log_file|awk '{print $3}'|sed 's/,//g'|sed '/^$/d' >> $output_dir/$RANK_ID/train_${CaseName}_loss.txt

  ActualLoss=`awk 'END {print}' $output_dir/$RANK_ID/train_${CaseName}_loss.txt`
  echo "Network = ${Network}" > $output_dir/$RANK_ID/${CaseName}.log
  echo "RankSize = ${RANK_SIZE}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "BatchSize = ${batch_size}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "DeviceType = ${DeviceType}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "CaseName = ${CaseName}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "ActualFPS = ${ActualFPS}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "TrainingTime = ${TrainingTime}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "ActualLoss = ${ActualLoss}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "E2ETrainingTime = ${e2e_time}" >> $output_dir/$RANK_ID/${CaseName}.log
  echo "CompileTime = ${CompileTime}" >> $output_dir/$RANK_ID/${CaseName}.log
  if [ $train_accuracy ]; then
    echo "TrainAccuracy = ${train_accuracy}" >> $output_dir/$RANK_ID/${CaseName}.log
  fi

  #eval版本需求开发中，精度结果临时看护最终的loss
  echo "Final Training Accuracy loss: $ActualLoss"
done
