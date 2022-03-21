#!/bin/bash

currentDir=$(cd "$(dirname "$0")";pwd)
export SET_RESULT_FILE=$currentDir/set_result.py
export RESULT_FILE=$currentDir/result.txt

function prepare() {
    # download dataset

    # verify  dataset

    # preprocess
    return 0
}

function exec_train() {

    # pytorch lenet5 sample
    #python3.7 $currentDir/pytorch_lenet5_train.py

    # tensorflow-1.15 wide&deep sample
    #python3.7 $currentDir/tensorflow_1_15_wide_deep.py
    #export ASCEND_DEVICE_ID=0
    cd $currentDir/test/

    bash train_performance_8p.sh --data_path=/npu/traindata/selfplay
    #result=$?
    # test sample
    #sleep 5
    #result=$?
    if [ $? != 0 ];then
        python3.7 $currentDir/set_result.py training "result" "NOK"
	exit 1
    else
        accuracy=`awk 'END {print $1}' $currentDir/test/output/7/MiniGo_ID0629_for_TensorFlow_bs128_8p_perf_loss.log`
        FPS=`grep FPS $currentDir/test/output/7/MiniGo_ID0629_for_TensorFlow_bs128_8p_perf.log |awk '{print $3}'`

        python3.7 $currentDir/set_result.py training "accuracy" $accuracy
        python3.7 $currentDir/set_result.py training "result" "OK"
        python3.7 $currentDir/set_result.py training "throughput_ratio" $FPS
    fi
}

function main() {

    prepare

    exec_train

}

main "$@"
ret=$?
exit $?
