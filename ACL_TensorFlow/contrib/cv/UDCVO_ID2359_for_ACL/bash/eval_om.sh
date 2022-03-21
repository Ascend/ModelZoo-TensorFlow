#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/eval_om.py \
--restore_path binfile/input \
--ground_truth_path testing/void_test_ground_truth_1500.txt \
--start_idx 0 \
--end_idx 800 \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--min_evaluate_z 0.2 \
--max_evaluate_z 5.0 \
--output_path binfile/output
