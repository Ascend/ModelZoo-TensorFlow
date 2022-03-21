#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../siamese_tf_128batch.om

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_x1/,$cur_dir/input_x2/ --output $cur_dir/output

#post process
python3 cal_accuracy.py --infer_result $cur_dir/output --ground_truth $cur_dir/ground_truth
