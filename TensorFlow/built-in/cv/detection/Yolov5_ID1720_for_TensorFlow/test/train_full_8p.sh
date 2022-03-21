#!/bin/bash
cur_path=`pwd`/../


#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
export JOB_ID=10000
#基础参数，需要模型审视修改
#Batch Size
batch_size=4
#网络名称，同目录名称
Network="Yolov5_ID1720_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=8
#训练epoch，可选
epochs=10
#训练step
train_steps=1e8
#学习率
learning_rate=0.002
learning_rate_end=0.00002

net_type='yolov5'
exec_mode='train'
eval_after_training=True

ckpt_path=
begin_epoch=0

steps=1e8
first_stage_epochs=0
warmup_epochs=3

precision_mode='allow_mix_precision'
#precision_mode='allow_fp32_to_fp16'
loss_scale_flag=0
loss_scale_value=256
overflow_dump=False

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_8p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi

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

cd $cur_path

wait

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait
output_dir=${output_dir:-"$cur_path/test/output/"}
echo output_dir=$output_dir
second_stage_epochs=${second_stage_epochs:-"$epochs"}

if [ $data_path ]; then
  data_classes_file=${data_classes_file:-"$data_path/labels.txt"}
  data_annotations_file=${data_annotations_file:-"$data_path/train_annotation.txt"}
fi

########## prepare environment ##########

export RANK_SIZE=$RANK_SIZE
export RANK_ID_START=0

BASE_PATH=`cd $(dirname $0); pwd`/../YOLOv5
echo "BASE_PATH="$BASE_PATH

export RANK_TABLE_FILE=$BASE_PATH/npu_config/8p.json

rm -rf /root/ascend/log
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

  rm -f data
  ln -s $BASE_PATH/data data

  export RANK_ID
  export DEVICE_ID=$RANK_ID
  export ASCEND_DEVICE_ID=$RANK_ID
  export DEVICE_INDEX=$RANK_ID

    corenum=`cat /proc/cpuinfo |grep 'processor' |wc -l`
    let a=RANK_ID*${corenum}/8
    let b=RANK_ID+1
    let c=b*${corenum}/8-1
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
  ${bind_core} python3 $BASE_PATH/train.py \
                              --net_type=$net_type \
                              --exec_mode=$exec_mode \
                              --eval_after_training=$eval_after_training \
                              --max_total_steps=$steps \
                              --npu_precision_mode=$precision_mode \
                              --npu_loss_scale_flag=$loss_scale_flag \
                              --npu_loss_scale_value=$loss_scale_value \
                              --npu_overflow_dump=$overflow_dump \
                              --data_classes_file=$data_classes_file \
                              --data_annotations_file=$data_annotations_file \
                              --train_batch_size=$batch_size \
                              --learning_rate_init=$learning_rate \
                              --learning_rate_end=$learning_rate_end \
                              --warmup_epochs=$warmup_epochs \
                              --first_stage_epochs=$first_stage_epochs \
                              --second_stage_epochs=$second_stage_epochs \
                              --initial_ckpt_path=$ckpt_path \
                              --begin_epoch=$begin_epoch \
                              2>&1 | tee $TMP_PATH/train_${RANK_ID}.log &

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
  FPS=`sed -e 's/\r/\n/g' $log_file | grep "train loss" | grep -a 'it/s' | tail -n 1 | sed -e 's/.* \([0-9]*\.*[0-9]*\)it\/s.*/\1/g'`
  FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${FPS}'}'`
  echo "Final Performance images/sec : $FPS"

  ############################## 精度结果处理 ##############################
  #精度计算，需要根据网络修改
  train_accuracy=`grep "Average Precision" $log_file | awk 'NR==1 {print $NF}'`
  if [ $train_accuracy ]; then
    echo "Final Training Accuracy mAP: $train_accuracy"
  fi

  ############################## 性能看护 ##############################

  Network=Yolov5_ID1720_for_TensorFlow

  DeviceType=`uname -m`
  CaseName=${Network}${name_bind}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
  ActualFPS=${FPS}
  TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

  # 提取Loss到train_${CaseName}_loss.txt中，需要根据模型修改
  sed -e 's/\r/\n/g' $log_file | grep -a 'train loss' | sed -e 's/.*train loss: \(-*[0-9]*\.*[0-9]*\).*/\1/g' > $output_dir/$RANK_ID/train_${CaseName}_loss.txt
  #最后一个迭代loss值，不需要修改
  ActualLoss=`awk 'END {print $1}' $output_dir/$RANK_ID/train_${CaseName}_loss.txt`   

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
  if [ $train_accuracy ]; then
    echo "ActualMAP = ${train_accuracy}" >> $output_dir/$RANK_ID/${CaseName}.log
  fi

  #eval版本需求开发中，精度结果临时看护最终的loss
  echo "Final Training Accuracy loss: $ActualLoss"
done

  echo "########## copying slog ##########"
  cp -r /root/ascend/log/ $output_dir/slog
  echo "########## DONE copying slog ##########"                     