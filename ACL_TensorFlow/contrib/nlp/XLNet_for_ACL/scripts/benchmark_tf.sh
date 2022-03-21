#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../XLNet_1batch.om

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_bins/input_ids,$cur_dir/input_bins/input_mask,$cur_dir/input_bins/segment_ids --output $cur_dir/npu_predict --outputSize 4

#post process
python3 cal_accuracy.py --infer_result $cur_dir/npu_predict --labels $cur_dir/input_bins/label_ids/


