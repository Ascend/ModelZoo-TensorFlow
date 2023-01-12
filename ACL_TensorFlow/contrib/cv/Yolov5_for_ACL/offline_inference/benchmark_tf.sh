#!/bin/bash

batch_size=$1

if [ X"$batch_size" = X ]; then
    batch_size=1
fi

run_script_file=$(readlink -f "$0")
run_script_dir=$(dirname $run_script_file)
root_dir=$(dirname $run_script_dir)

model_name=Yolov5
# 已下参数请根据实际路径修改
data_dir=$root/input_bins  # 需要先把图片转成bin
om_file=$root_dir/yolov5_tf2_gpu.om
output_dir=$root_dir/offline_inference/output_bins

rm -rf $output_dir
mkdir -p $output_dir

$root_dir/Benchmark/out/benchmark \
--om $om_file \
--modelType $model_name \
--dataDir $data_dir \
--outDir $output_dir \
--batchSize $batch_size \
--imgType bin \
--useDvpp 0