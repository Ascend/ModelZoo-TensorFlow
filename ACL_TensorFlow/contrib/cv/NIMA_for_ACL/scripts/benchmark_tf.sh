#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../nima_1batch_input_fp16_output_fp32.om
batchsize=1
model_name=NIMA
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/xdata --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
python3 SSRC_compute.py $cur_dir/$output_dir $cur_dir/input_bins/ydata
