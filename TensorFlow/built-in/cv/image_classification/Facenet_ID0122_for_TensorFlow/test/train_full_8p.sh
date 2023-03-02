#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export JOB_ID=10096
export RANK_TABLE_FILE=${cur_path}/test/ranktable_8p.json
export ENABLE_FORCE_V2_CONTROL=1

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

#export ASCEND_DEVICE_ID=7
#基础参数，需要模型审视修改
#Batch Size
batch_size=90
#train epoch number
train_epoch=12
#网络名称，同目录名称
Network="Facenet_ID0122_for_TensorFlow"
#Device数量，单卡默认为1
export RANK_SIZE=8
RANK_SIZE=8

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_full_8p.sh "
	exit 1
fi

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
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
	nohup python3 ${cur_path}/src/train_softmax.py \
		--logs_base_dir ${cur_path}/src/logs/$ASCEND_DEVICE_ID \
		--models_base_dir ${cur_path}/src/models/$ASCEND_DEVICE_ID \
		--data_dir ${data_path}/CASIA-WebFace_182/ \
		--batch_size ${batch_size} \
		--image_size 160 \
		--model_def models.inception_resnet_v1 \
		--optimizer ADAM \
		--learning_rate -1 \
		--max_nrof_epochs ${train_epoch} \
		--keep_probability 0.8 \
		--random_crop \
		--random_flip \
		--random_rotate \
		--use_fixed_image_standardization \
		--learning_rate_schedule_file ${cur_path}/data/learning_rate_schedule_classifier_casia_8p.txt \
		--weight_decay 5e-4 \
		--embedding_size 512 \
		--validation_set_split_ratio 0.05 \
		--validate_every_n_epochs 5 \
		--prelogits_norm_loss_factor 5e-4 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

echo "Final Training Duration sec : $e2etime"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep  RegLoss $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| tail -n +2 | head -2999 | awk '{print $4}' | tr -d s | awk '{sum+=$1} END {print sum/NR}'`
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
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视

grep RegLoss $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk  '{print $6}' | tr -d ' ' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#打印，不需要修改

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

unset RANK_TABLE_FILE
unset RANK_SIZE
ckpt_path=`ls ${cur_path}/src/models/$ASCEND_DEVICE_ID/*/*ckpt-${train_epoch}.index`
ckpt_path=${ckpt_path%.*}
# run evalute
python3 ${cur_path}/src/train_softmax.py \
    --logs_base_dir ${cur_path}/src/logs/ \
	--models_base_dir ${cur_path}/src/models/ \
	--data_dir ${data_path}/CASIA-WebFace_182/ \
	--lfw_dir ${cur_path}/lfw/datasets \
	--pretrained_model ${ckpt_path} \
	--batch_size ${batch_size} \
	--image_size 160 \
	--epoch_size 1 \
	--model_def models.inception_resnet_v1 \
	--optimizer ADAM \
	--learning_rate -1 \
	--max_nrof_epochs 1 \
	--keep_probability 0.8 \
	--random_crop \
	--random_flip \
	--random_rotate \
	--use_fixed_image_standardization \
	--learning_rate_schedule_file ${cur_path}/data/learning_rate_schedule_classifier_casia.txt \
	--weight_decay 5e-4 \
	--embedding_size 512 \
	--validation_set_split_ratio 0 \
	--validate_every_n_epochs 5 \
	--prelogits_norm_loss_factor 5e-4 >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1

#train_accuracy
train_accuracy=`grep 'Accuracy:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $2}' |tail -1`

#打印，不需要修改
echo "train_accuracy : $train_accuracy"

RANK_SIZE=8

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