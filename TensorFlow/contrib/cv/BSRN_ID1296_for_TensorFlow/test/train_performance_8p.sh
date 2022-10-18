#!/bin/bash

##########################################################
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
##########################################################
# shell脚本所在路径
cur_path=`echo $(cd $(dirname $0);pwd)`

# 当前执行网络的名称
Network=`echo $(cd $(dirname $0);pwd) | awk -F"/" '{print $(NF-1)}'`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
RANK_ID_START=0
batch_size=8
export JOB_ID=10087
export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
# 路径参数初始化
data_path=""


#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#export RANK_ID=npu8p
export SLOG_PRINT_TO_STDOUT=0

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
	  --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
    "
    exit 1
fi

# 参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
        train_steps=`echo ${para#*=}`
	elif [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi



# 设置打屏日志文件名，请保留，文件名为${print_log}
#print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
#etp_flag=${etp_running_flag}
##if [ x"${etp_flag}" != x ];
##then
#    #echo "running without etp..."
#    #print_log_name=`ls /home/ma-user/modelarts/log/ | grep proc-rank`
#    #print_log="/home/ma-user/modelarts/log/${print_log_name}"
##fi
#echo ${print_log}
#
#CaseName=""
#function get_casename()
#{
#    if [ x"${perf_flag}" = x1 ];
#    then
#        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
#    else
#        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
#    fi
#}

# 跳转到code目录
cd ${cur_path}/../
rm -rf ${cur_path}/../checkpoints
#rm -rf ./test/output/${ASCEND_DEVICE_ID}
#mkdir -p ./test/output/${ASCEND_DEVICE_ID}

# 训练开始时间记录，不需要修改
start_time=$(date +%s)
max_steps=4320
save_freq=$[$max_steps/2]
##########################################################
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
#########第3行 至 100行，请一定不要、不要、不要修改##########
##########################################################

#=========================================================
#=========================================================
#========训练执行命令，需要根据您的网络进行修改==============
#=========================================================
#=========================================================
# 您的训练数据集在${data_path}路径下，请直接使用这个变量获取
# 您的训练输出目录在${output_path}路径下，请直接使用这个变量获取
# 您的其他基础参数，可以自定义增加，但是batch_size请保留，并且设置正确的值
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
	  DEVICE_INDEX=$RANK_ID
    export DEVICE_INDEX=${DEVICE_INDEX}

    #创建DeviceID输出目录，不需要修改
    if [ -d $cur_path/output/$ASCEND_DEVICE_ID ];then
        rm -rf $cur_path/output/$ASCEND_DEVICE_ID
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
    fi
    ## 校验是否传入output_path,不需要修改
    output_path="./test/output/${ASCEND_DEVICE_ID}"
    print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
        #执行训练脚本，需要模型审视修改
    relative_path_LR="DIV2K/DIV2K_train_LR_bicubic"
    relative_path_HR="DIV2K/DIV2K_train_HR"
nohup python3.7 ./train.py \
    --data_input_path=${data_path}/${relative_path_LR}\
    --data_truth_path=${data_path}/${relative_path_HR} \
    --train_path=./checkpoints \
    --chip='npu' \
    --model='bsrn' \
    --dataloader='div2k_loader' \
    --batch_size=8 \
    --max_steps=$max_steps \
    --save_freq=$save_freq \
    --scales='4' 1>${print_log} 2>&1 &
done
wait

# if [ $ASCEND_DEVICE_ID -eq $[7] ];then
    relative_path_LR="BSD100/LR"
    relative_path_HR="BSD100/SR"
    # after training, load the model to check the performance
    str1="model.ckpt-"
    str2=$[$max_steps/$RANK_SIZE]
    relative_path_checkpoint=${str1}${str2}

    python3.7 ./validate_bsrn.py \
    --dataloader=basic_loader \
    --data_input_path=${data_path}/${relative_path_LR} --data_truth_path=${data_path}/${relative_path_HR} \
    --restore_path=./checkpoints/${relative_path_checkpoint}  \
    --model=bsrn \
    --scales='4' \
    --save_path=./result-pictures 1>>${print_log} 2>&1
# fi
# 性能相关数据计算
StepTime=`grep "sec/batch" ${print_log} | tail -n 20 | awk '{print $(NF-2)}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${StepTime}'}'`

# 精度相关数据计算
#train_accuracy=`grep "Final Accuracy accuracy" ${print_log}  | awk '{print $NF}'`
PSNR=`grep "Final PSNR" ${print_log} | awk '{print $NF}'`
SSIM=`grep "Final SSIM" ${print_log} | awk '{print $NF}'`
# 提取所有loss打印信息
grep "loss" ${print_log} | awk -F ":" '{print $4}'| grep "loss" |awk -F "," '{print $3}'|awk '{print $2}' > ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt


###########################################################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
#########后面的所有内容请不要修改###########################
###########################################################

# 判断本次执行是否正确使用Ascend NPU
use_npu_flag=`grep "The model has been compiled on the Ascend AI processor" ${print_log} | wc -l`
if [ x"${use_npu_flag}" == x0 ];
then
    echo "------------------ ERROR NOTICE START ------------------"
    echo "ERROR, your task haven't used Ascend NPU, please check your npu Migration."
    echo "------------------ ERROR NOTICE END------------------"
else
    echo "------------------ INFO NOTICE START------------------"
    echo "INFO, your task have used Ascend NPU, please check your result."
    echo "------------------ INFO NOTICE END------------------"
fi

# 获取最终的casename，请保留，case文件名为${CaseName}
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'

# 重命名loss文件
if [ -f ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ];
then
    mv ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ./test/output/${ASCEND_DEVICE_ID}/${CaseName}_loss.txt
fi

# 训练端到端耗时
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
# 输出性能FPS/单step耗时/端到端耗时
echo "Final Performance images/sec : $FPS"
echo "Final Performance sec/step : $StepTime"
echo "E2E Training Duration sec : $e2e_time"

# 输出训练精度
#echo "Final Train Accuracy : ${train_accuracy}"
echo "Final Train Accuracy : ${PSNR}"
echo "Final SSIM : ${SSIM}"
# 最后一个迭代loss值，不需要修改
ActualLoss=(`awk 'END {print $NF}' $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = `uname -m`" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${StepTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
