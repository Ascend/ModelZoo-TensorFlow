#!/usr/bin/env sh

python eval.py \
--multitask True \
--model_type LMTCNN-1-1 \
--eval_dir ./tfrecord/train_val_test_per_fold_agegender/test_fold_is_2 \
--eval_data test \
--model_dir ./models/train_val_test_per_fold_agegender/test_fold_is_2/LMTCNN-2-1-run-29437 \
--result_dir ./results \
--batch_size 128