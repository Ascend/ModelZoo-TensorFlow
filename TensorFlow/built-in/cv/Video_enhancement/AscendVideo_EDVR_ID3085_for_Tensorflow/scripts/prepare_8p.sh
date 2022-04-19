#!/bin/bash
cur_dir=`pwd`
root_dir=${cur_dir}

mkdir data
for i in $(seq 0 7)
do
    if [ ! -d "D$i" ];then
        bash scripts/create_new_experiment.sh D${i}
    fi
done
