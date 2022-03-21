#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../transnet_tf_1batch.om
batchsize=1
model_name=transnet
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
python3 video_pre_postprocess.py --video_path $cur_dir/BigBuckBunny.mp4 --predict_path $cur_dir/$output_dir/$model_name --mode postprocess
