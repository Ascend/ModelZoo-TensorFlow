#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../dpn_8batch.om

#start offline inference
python3 npu_inference.py ./input_bins/ ./npu_output/ ${om_name}

