#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../DIEN_128batch.om
batchsize=128
model_name=DIEN
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/mid_his/,$cur_dir/cat_his/,$cur_dir/uids/,$cur_dir/mids/,$cur_dir/cats/,$cur_dir/mid_mask/,$cur_dir/sl/ --output $cur_dir/$output_dir

#post process
python3 post_process.py $cur_dir/target/ $cur_dir/$output_dir
