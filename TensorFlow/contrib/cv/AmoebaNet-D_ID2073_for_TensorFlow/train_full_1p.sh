#! /bin/bash

DATA_DIR=/home/test_user03/tf_records/
MODEL_DIR=/home/test_user03/xx


nohup python3 amoeba_net.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --num_cells=6 \
  --image_size=224 \
  --num_epochs=35 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --mode=train_and_eval \
  --iterations_per_loop=1251 \
  > train_full_1p.log 2>&1 &

