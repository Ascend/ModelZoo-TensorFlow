#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../econet_ucf101_4batch.om

#start offline inference
${benchmark_dir}/benchmark --model ${om_name} --input ${cur_dir}/input_bins --output ${cur_dir}/npu_predict

#post process
python3 cal_accuracy.py --label_path ${cur_dir}/input_bins/ucf101_label.pkl --output_path ${cur_dir}/npu_predict

