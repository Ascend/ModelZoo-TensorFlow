#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../Centernet_2batch_input_fp16_output_fp32.om
batchsize=1
model_name=CenterNet
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0 --outputMem 2400

#post process
mkdir npu_predict
python3 bin2txt.py $cur_dir/$output_dir/$model_name ./npu_predict/
cd centernet_postprocess
python3 pascalvoc.py --detfolder ../npu_predict/
