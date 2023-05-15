#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export JOB_ID=10096
export RANK_TABLE_FILE=${cur_path}/test/ranktable_8p.json
export ENABLE_FORCE_V2_CONTROL=1

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=90
#train epoch number
train_epoch=15
#网络名称，同目录名称
Network="Facenet_ID0122_for_TensorFlow"
#Device数量，单卡默认为1
export RANK_SIZE=8
RANK_SIZE=8

#参数配置
data_path=""
switch_config="${cur_path}/switch_config.txt"

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_full_8p.sh "
	exit 1
fi

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
  elif [[ $para == --bind_core* ]];then
    bind_core=`echo ${para#*=}` 
    name_bind="_bindcore"
	elif [[ $para == --switch_config* ]];then
		switch_config=`echo ${para#*=}`
	fi
done

if [[ $data_path  == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi
##############执行训练##########
cd $cur_path
rm -rf $cur_path/test/output/*

start=$(date +%s)

for((ID=0; ID<$RANK_SIZE;ID++));
do
	export ASCEND_DEVICE_ID=${ID}
	export RANK_ID=${ID}
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
	mkdir -p $cur_path/src/logs/$ASCEND_DEVICE_ID
	rm -rf ${cur_path}/src/models/$ASCEND_DEVICE_ID
	mkdir -p ${cur_path}/src/models/$ASCEND_DEVICE_ID
  # 绑核，不需要的绑核的模型删除，需要的模型审视修改
  let a=RANK_ID*12
  let b=RANK_ID+1
  let c=b*12-1

  corenum=`cat /proc/cpuinfo |grep 'processor' | wc -l`
  let a=RANK_ID*${corenum}/8
  let b=RANK_ID+1
  let c=b*${corenum}/8-1
  if [ "x${bind_core}" != x ];then
      bind_core="taskset -c $a-$c"
  fi

  nohup ${bind_core} python3 ${cur_path}/src/train_softmax.py \
		--logs_base_dir ${cur_path}/src/logs/$ASCEND_DEVICE_ID \
		--models_base_dir ${cur_path}/src/models/$ASCEND_DEVICE_ID \
		--data_dir ${data_path}/CASIA-WebFace_182/ \
		--lfw_dir ${data_path}/lfw_mtcnnpy_160/ \
		--batch_size ${batch_size} \
		--image_size 160 \
		--model_def models.inception_resnet_v1 \
		--optimizer ADAM \
		--learning_rate 0.6 \
		--learning_rate_decay_epochs 1 \
		--learning_rate_decay_factor 0.7 \
		--max_nrof_epochs ${train_epoch} \
		--keep_probability 1.0 \
		--random_crop \
		--random_flip \
		--lfw_distance_metric 1 \
		--lfw_use_flipped_images \
		--lfw_subtract_mean \
		--use_fixed_image_standardization \
		--learning_rate_schedule_file ${cur_path}/data/learning_rate_schedule_classifier_casia_8p.txt \
		--switch_config ${switch_config} \
		--weight_decay 5e-4 \
		--embedding_size 512 \
		--validation_set_split_ratio 0.05 \
		--validate_every_n_epochs 5 \
		--prelogits_norm_loss_factor 1e-3 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep  RegLoss $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $4}' | tr -d s | awk '{sum+=$1} END {print sum/NR}'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`
FPS=$(awk 'BEGIN{print '$FPS'*8}')
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#吞吐量
ActualFPS=${FPS}
###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视

grep RegLoss $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk  '{print $6}' | tr -d ' ' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#打印，不需要修改

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#train_accuracy
train_accuracy=`grep -rn 'Accuracy:' $cur_path/test/output/*/* | awk '{print $2}' | awk -F'%' 'BEGIN {max = 0} {if ($1+0 > max+0) max=$1} END {print max}'`
train_accuracy=${train_accuracy%%+*}
#打印，不需要修改
echo "train_accuracy : $train_accuracy"

saved_model_path=${cur_path}/src/models/saved_model_path/`date '+%Y%m%d%H%M'`
mkdir $saved_model_path -p
bst_d=`grep -rn $train_accuracy $cur_path/test/output/*/* | head -n 1`
bst_d=${bst_d%/*}
bst_d=${bst_d##*/}
bst_e=`grep -rn -B 10 $train_accuracy $cur_path/test/output/*/* | head -n 1 | awk '{print $2}'`
bst_e=${bst_e%%]*}
bst_e=${bst_e#*[}
cp ${cur_path}/src/models/${bst_d}/*/*ckpt-${bst_e}* ${saved_model_path}
cp ${cur_path}/src/models/${bst_d}/*/*.meta ${saved_model_path}
cp ${cur_path}/src/models/${bst_d}/*/checkpoint ${saved_model_path}
sed -i "1s/ckpt-${train_epoch}/ckpt-${bst_e}/" ${saved_model_path}/checkpoint
echo "saved model path: ${saved_model_path}/"


##获取错误信息
#系统错误信息
#error_msg="PrintOp Failed to compile node \[name gradients\/InceptionResnetV1\/Bottleneck\/BatchNorm\/cond\/FusedBatchNormV3_1_grad\/FusedBatchNormGradV3_Update, type BNTrainingUpdateGrad\]"
#判断错误信息是否和历史状态一致，此处无需修改
#Status=`grep "${error_msg}" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
#ModelStatus="图执行FAIL"
#DTS单号或者issue链接
#DTS_Number="DTS2021080623275"


#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RANK_SIZE = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log