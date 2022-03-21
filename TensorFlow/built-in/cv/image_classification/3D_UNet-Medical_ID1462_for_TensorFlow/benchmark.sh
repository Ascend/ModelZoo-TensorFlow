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

    # test sample

    cd $currentDir/test/
    bash train_full_1p.sh
    steps_sec=`grep "global_step/sec"  $currentDir/test/output/1/train_1.log|awk '{print $2}'| awk '{line[NR]=$0} END {for(i=3;i<=NR;i++) print line[i]}'|awk '{sum+=$1} END {print sum/NR}'`
    FPS=`echo "${steps_sec} 2" | awk '{printf("%.4f\n",$1*$2)}'`
    train_accuracy=`grep "iou =" $currentDir/test/output/1/train_1.log|awk 'END {print $12}'|cut -d , -f 1`
    python3.7 $currentDir/set_result.py training "accuracy" $train_accuracy
    python3.7 $currentDir/set_result.py training "result" "NOK"
    python3.7 $currentDir/set_result.py training "throughput_ratio" $FPS
}

function main() {

    prepare

    exec_train

}

main "$@"
ret=$?
exit $?
