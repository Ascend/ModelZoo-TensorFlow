export JOB_ID=10086
export ASCEND_DEVICE_ID=2

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
BookCorpus_DIR=/home/TestUser03/bupt_lil/Data/BookCorpus
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/pretrain

#start exec
python3.7 run_pretraining.py \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_warmup_steps=100 \
  --num_train_steps=100000 \
  --optimizer_type=adam \
  --manual_fp16=True \
  --use_fp16_cls=True \
  --input_files_dir=$BookCorpus_DIR/pretrain_data \
  --eval_files_dir=/home/TestUser03/bupt_lil/Data/eval_data \
  --npu_bert_debug=False \
  --npu_bert_use_tdt=True \
  --do_train=True \
  --num_accumulation_steps=1 \
  --npu_bert_job_start_file= \
  --iterations_per_loop=100 \
  --save_checkpoints_steps=10000 \
  --npu_bert_clip_by_global_norm=False \
  --distributed=False \
  --npu_bert_loss_scale=0 \
  --output_dir=$OUT_DIR \
  --out_log_dir=$OUT_DIR/loss
