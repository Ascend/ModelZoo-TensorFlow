#!/bin/bash

nohup python3 -u ../part_seg/test.py \
    > ../part_seg/test_result.log 2>&1 &
