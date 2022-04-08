#!/bin/bash

python3 /home/work/user-job-dir/mytask/freeze_graph.py \
        --h5_obs_path='obs://kyq/mytestnew/mytestnew1h5/tcn.h5' \
        --h5_path='/cache/h5/tcn.h5' \
        --pb_obs_path='obs://kyq/my/tcn.pb'
