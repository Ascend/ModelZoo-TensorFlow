#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../Roformer_1batch.om
batchsize=1
model_name=Roformer
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_bins/Input-Segment,$cur_dir/input_bins/Input-Token --output $cur_dir/$output_dir

#post process
python3 run_pb_inference.py --model $cur_dir/../Roformer.pb --input $cur_dir/input_bins --npu_output $cur_dir/$output_dir --batchsize $batchsize
