#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/Benchmark/out


om_name=$cur_dir/pb_model/rpn_x600.om
output_dir='results_rpn_x600'
rm -rf $cur_dir/$output_dir/*
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_bins_x600 --output $cur_dir/$output_dir

om_name=$cur_dir/pb_model/rpn_x800.om
output_dir='results_rpn_x800'
rm -rf $cur_dir/$output_dir/*
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_bins_x800 --output $cur_dir/$output_dir