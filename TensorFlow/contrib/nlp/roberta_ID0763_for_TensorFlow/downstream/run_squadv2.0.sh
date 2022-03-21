export JOB_ID=10086
export ASCEND_DEVICE_ID=0

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
SQUAD_DIR=/home/TestUser03/bupt_lil/Data/SQuAD/data
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/v2-512

#rm -rf output

python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/tf_model/roberta_large.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=4 \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --version_2_with_negative=True


python3 $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $OUT_DIR/predictions.json \
  --na-prob-file $OUT_DIR/null_odds.json