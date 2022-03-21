#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../ssd_inceptionv2_tf.om
batchsize=1
model_name=ssd_inceptionv2
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --model ${om_name} --input ${cur_dir}/input_bins --output ${output_dir} --outputSize "800,200,2,200"

#post process
python3 coco_afterProcess.py --result_file_path=${output_dir} --img_conf_path=${cur_dir}/img_info --save_json_path=${cur_dir}/result.json
python3 eval_coco.py ${cur_dir}/result.json ${cur_dir}/instances_minival2014.json
