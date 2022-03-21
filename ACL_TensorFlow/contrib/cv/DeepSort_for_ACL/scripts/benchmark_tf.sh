#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out

#start offline inference
python3 generate_detections_npu.py --model=../mars-small128.pb --mot_dir=dataset/test/ --output_dir=npu_result/


