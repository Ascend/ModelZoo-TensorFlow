#!/bin/bash

nohup python3 -u ../part_seg/train.py \
    > ../part_seg/train_result.log 2>&1 &
