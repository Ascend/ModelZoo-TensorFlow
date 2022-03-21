#!/usr/bin/env bash

export install_path_atc=/usr/local/Ascend
export install_path_acllib=/usr/local/Ascend
export install_path_toolkit=/usr/local/Ascend
export install_path_opp=/usr/local/Ascend
export driver_path=/usr/local/Ascend
export ASCEND_OPP_PATH=/usr/local/Ascend
export PATH=/usr/local/python3.7.5/bin:${install_path_atc}/atc/ccec_compiler/bin:${install_path_atc}/atc/bin:$PATH
export PYTHONPATH=${install_path_atc}/atc/python/site-packages/:${install_path_atc}/atc/python/site-packages/auto_tune.egg:${install_path_atc}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path_acllib}/acllib/lib64:${install_path_atc}/atc/lib64:${install_path_toolkit}/toolkit/lib64:${driver_path}add-ons:$LD_LIBRARY_PATH

batch_size=1268
len=123
soc_version=Ascend310
cur_dir=`pwd`
out_path=${cur_dir}

atc --model=./nmt.pb --framework=3 --output=${out_path}/"nmt_${batch_size}" --soc_version=${soc_version} --input_shape="src_ids:${batch_size},${len};src_seq_len:${batch_size}" --out_nodes='dynamic_seq2seq/decoder/decoder/TensorArrayStack_1/TensorArrayGatherV3:0'