#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/din_512batch_dynamic_shape.om
batchsize=512
model_name=din
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --model $om_name --input ./input_bins/pl2,./input_bins/pl3,./input_bins/pl4,./input_bins/pl5, --dymConfig dataset_conf.txt --output $output_dir 

#post process
python3 post_process.py ./ ./results  > acc.txt
test_gauc=`grep "test_gauc:" acc.txt |awk '{print $2}'`
test_auc=`grep "test_auc:" acc.txt |awk '{print $4}'`
echo "test_gauc: ${test_gauc}"
echo "test_auc: ${test_auc}"