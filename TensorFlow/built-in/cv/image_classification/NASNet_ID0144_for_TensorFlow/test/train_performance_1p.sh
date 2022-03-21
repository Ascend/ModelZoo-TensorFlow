#!/bin/bash
cd ..

mkdir -p ./test/output/$ASCEND_DEVICE_ID
rm -rf ./test/output/$ASCEND_DEVICE_ID/*

begin=$(date +%s)
nohup python3 ./train_image_classifier.py --train_dir=./test/output/checkpoints --dataset_name=cifar10 --dataset_split_name=train --dataset_dir=./data --model_name=nasnet_large --log_every_n_steps=1 --max_number_of_steps=20 2>&1 >./test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
end=$(date +%s)

e2etime=$(( $end - $begin ))
step_sec=`grep -a 'global step' ./test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'END {print $11}' | cut -c 2-`

echo "-------- Final Result --------"
echo "Final Performance s/step : $step_sec"
echo "Final Training Duration sec : $e2etime"