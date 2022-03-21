#!/bin/bash
cur_dir=$(cd "$(dirname "$0")"; pwd)
echo "$cur_dir"
rm -rf $cur_dir/../model_params.json
cp $cur_dir/../model_params_acc.json $cur_dir/../model_params.json

python3 main.py
