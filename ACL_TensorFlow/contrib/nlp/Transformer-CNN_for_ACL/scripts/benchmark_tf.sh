#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out


om_name=$cur_dir/../encoder.om
output_dir='results_encoder'
rm -rf $cur_dir/$output_dir/*
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_bins/input_x0,$cur_dir/input_bins/input_x1 --output $cur_dir/$output_dir


om_name=$cur_dir/../model.om
output_dir='results'
rm -rf $cur_dir/$output_dir/*
#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/results_encoder --output $cur_dir/$output_dir

#post process
python3 postprocess.py --infer_result $cur_dir/results --ground_truth $cur_dir/data/test.csv