#!/bin/bash


cur_path=`echo $(cd $(dirname $0);pwd)`


perf_flag=`echo $0 | grep performance | wc -l`


Network=`echo $(cd $(dirname $0);pwd) | awk -F"/" '{print $(NF-1)}'`

export RANK_SIZE=1
export RANK_ID=0
export JOB_ID=10087


data_path=""
output_path=""


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


if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi


if [[ $output_path == "" ]];then
    output_path="./test/output/${ASCEND_DEVICE_ID}"
fi


print_log="./test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log"
modelarts_flag=${MODELARTS_MODEL_PATH}
if [ x"${modelarts_flag}" != x ];
then
    echo "running without etp..."
    print_log_name=`ls /home/ma-user/modelarts/log/ | grep proc-rank`
    print_log="/home/ma-user/modelarts/log/${print_log_name}"
fi
echo "### get your log here : ${print_log}"

CaseName=""
function get_casename()
{
    if [ x"${perf_flag}" = x1 ];
    then
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
    else
        CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'acc'
    fi
}


cd ${cur_path}/../
rm -rf ./test/output/${ASCEND_DEVICE_ID}
mkdir -p ./test/output/${ASCEND_DEVICE_ID}


start_time=$(date +%s)
##########################################################

##########################################################

#=========================================================
#=========================================================

pwd



python -m pip install --upgrade pip

pip install keras==2.2.4

#cp -r  ./mat/*  /usr/local/Ascend/ascend-toolkit/5.1.RC2.alpha001/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/


pip3 list | grep -i keras

ls ./h5
mkdir -p  ~/.keras/models/
#ls ~
cp -r   ./h5/*   ~/.keras/models/
ls ~/.keras/models/

#export ASCEND_GLOBAL_LOG_LEVEL=0
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#find /home/ma-user -iname 'libgomp.so.1'
unset LD_PRELOAD
export LD_PRELOAD=/home/ma-user/miniconda3/envs/TensorFlow-1.15-arm/bin/../lib/libgomp.so.1:$LD_PRELOAD

batch_size=16

if [ x"${modelarts_flag}" != x ];
then


    python3.7 ./train.py \
        --input_dir=${data_path} \
        --output_dir=${output_path} \
        --batch_size=16 \
        --epochs=500 \
        --number_of_images=8000 \
        --train_test_ratio=0.8 \
        --model_save_dir='./model/'
else


    python3.7 ./train.py \
        --input_dir=${data_path} \
        --output_dir=${output_path} \
        --batch_size=16 \
        --epochs=500 \
        --number_of_images=8000 \
        --train_test_ratio=0.8 \
        --model_save_dir='./model/'
         > ${print_log}
fi




StepTime=`grep "sec/step :" ${print_log} | tail -n 10 | awk '{print $NF}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${StepTime}'}'`


train_accuracy=`grep "Final Accuracy accuracy" ${print_log}  | awk '{print $NF}'`

grep "loss :" ${print_log} | awk -F ":" '{print $4}' | awk -F "-" '{print $1}' > ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt


###########################################################

###########################################################


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


get_casename


if [ -f ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ];
then
    mv ./test/output/${ASCEND_DEVICE_ID}/my_output_loss.txt ./test/output/${ASCEND_DEVICE_ID}/${CaseName}_loss.txt
fi


end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"

echo "Final Performance images/sec : $FPS"
echo "Final Performance sec/step : $StepTime"
echo "E2E Training Duration sec : $e2e_time"


echo "Final Train Accuracy : ${train_accuracy}"


ActualLoss=(`awk 'END {print $NF}' $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`)


echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = `uname -m`" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${StepTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log