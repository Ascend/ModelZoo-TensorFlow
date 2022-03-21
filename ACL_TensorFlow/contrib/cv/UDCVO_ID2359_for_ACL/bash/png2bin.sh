#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/png2bin.py \
--image_path testing/void_test_image_1500.txt \
--interp_depth_path testing/void_test_interp_depth_1500.txt \
--validity_map_path testing/void_test_validity_map_1500.txt \
--ground_truth_path testing/void_test_ground_truth_1500.txt \
--start_idx 0 \
--end_idx 800 \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--output_path binfile/output \
--n_thread 4
