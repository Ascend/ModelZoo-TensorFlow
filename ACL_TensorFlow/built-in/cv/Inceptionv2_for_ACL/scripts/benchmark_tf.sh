#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../inceptionv2_tf_1batch.om
batchsize=1
model_name=inceptionv2
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
python3 $cur_dir/imagenet_accuarcy_cal.py --infer_result $cur_dir/$output_dir/$model_name --label $cur_dir/ILSVRC2012val-label-index.txt --offset 1
