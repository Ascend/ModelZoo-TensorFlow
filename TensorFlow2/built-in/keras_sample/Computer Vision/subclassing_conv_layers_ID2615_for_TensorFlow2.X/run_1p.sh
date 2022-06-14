#!/bin/bash
data_path=""
nohup python3 subclassing_conv_layers.py --epochs=2 --batch_size=256 --data_path=$data_path >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &