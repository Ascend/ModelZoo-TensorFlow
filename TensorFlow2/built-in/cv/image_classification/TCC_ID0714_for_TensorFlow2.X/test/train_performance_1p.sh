#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`
#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087

RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="TCC_ID0714_for_TensorFlow2.X"
batch_size=2

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./performance.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode       precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		   data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_path		         source data of training
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ci_cp* ]];then
        ci_cp=`echo ${para#*=}`
    fi
done

if [[ $ci_cp == "1" ]];then
    cp -r $data_path ${data_path}_bak
fi

#data_path='../tcc/'
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
cd $cur_path/../
mkdir -p logs/checkpoints/

cd tcc
sed -i "s|tmp|${data_path}/tmp|g" train.py
sed -i "s|tmp|${data_path}/tmp|g" config.py
sed -i "s|CONFIG.TRAIN.MAX_ITERS = 150000|CONFIG.TRAIN.MAX_ITERS = 1000|g" config.py
cd configs
sed -i "s|tmp|${data_path}/tmp|g" demo.yml

cd ../../test

#cp -r ${data_path}/tmp/alignment_logs ${data_path}/tmp/alignment_logs.bak
#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

#    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
#    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
#    nohup python3.7 $cur_path/../train_adda_seg.py ${data_path}/data/inria_test source_image source_label_index target_image adda_deeplab_v3p.h5 \
#        --optimizer adam \
#		--base_learning_rate 1e-4 \
#		--min_learning_rate 1e-7 \
#		--image_width 256 \
#		--image_height 256 \
#		--image_channel 3 \
#		--image_suffix .png \
#		--label_suffix .png \
#		--n_class 2 \
#		--batch_size 2 \
#		--iterations 50 \
#		--weight_decay 1e-4 \
#		--initializer he_normal \
#		--bn_epsilon 1e-3 \
#		--bn_momentum 0.99 \
#		--pre_trained_model ./logs/checkpoints/deeplab_v3p_base.h5 \
#		--source_fname_file ${data_path}/data/inria_test/source.txt \
#		--target_fname_file ${data_path}/data/inria_test/target.txt \
#		--logs_dir ./logs \
#		--augmentations flip_x,flip_y,random_crop \
#		--display 1 \
#		--snapshot 5   > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

	#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 $cur_path/../tcc/train.py \
        --precision_mode=${precision_mode} \
        --over_dump=${over_dump} \
        --over_dump_path=${over_dump_path} \
        --data_dump_flag=${data_dump_flag} \
        --data_dump_step=${data_dump_step} \
        --data_dump_path=${data_dump_path} \
	      --profiling=${profiling} \
	      --profiling_dump_path=${profiling_dump_path}\
		    --alsologtostderr \
        --force_train   > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

    # sleep 60
    # num=`grep 'E19999' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | wc -l`
    # while [ ${num} -eq 0 ]
    # do
    #     num=`grep 'E19999' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | wc -l`
    #     sleep 5
    # done
    # ps -ef | grep python3 | grep tcc | grep train.py | awk '{system("kill -9 "$2)}'
    # echo 'killed TCC_ID0714_for_TensorFlow2.X' >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
done
wait

cd ../tcc
sed -i "s|${data_path}/tmp|tmp|g" train.py
sed -i "s|${data_path}/tmp|tmp|g" config.py
# sed -i "s|CONFIG.TRAIN.MAX_ITERS = 1500|CONFIG.TRAIN.MAX_ITERS = 15000|g" config.py
cd ../test
#rm -rf ${data_path}/tmp/alignment_logs
#mv ${data_path}/tmp/alignment_logs.bak ${data_path}/tmp/alignment_logs
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#step_per_s=`grep 'global_step/sec' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
step_per_s=`grep 's/iter' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $6}'|awk -F "s" 'END{print $1}'`
step_per_s=`awk 'BEGIN{printf "%.2f\n",1/'${step_per_s}'}'`
# FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${RANK_SIZE}'*'${step_per_s}'}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${step_per_s}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep 'accuracy =' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $9}'|awk -F "," '{print $1}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Loss:' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $8}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = None" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log

sed -i "/AttributeError/d" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log

if [[ $ci_cp == "1" ]];then
    rm -rf $data_path
    mv ${data_path}_bak $data_path
fi
