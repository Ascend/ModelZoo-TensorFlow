#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../pwcnet_1batch.om

#start offline inference
${benchmark_dir}/benchmark --model ${om_name} --input ${cur_dir}/input_bins/image --output ${cur_dir}/npu_predict

#Post process
python3 evaluation.py --gt_path ${cur_dir}/input_bins/gt --output_path ${cur_dir}/npu_predict

