export JOB_ID=10086
export ASCEND_DEVICE_ID=0

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
SQUAD_DIR=/home/TestUser03/bupt_lil/Data/SQuAD/data
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/v1-384

#rm -rf output

python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/tf_model/roberta_large.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=4 \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --version_2_with_negative=False


python3 $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUT_DIR/predictions.json