BASE_DIR=/home/test_user02/lil/output/msame/squadv2
BERT_BASE_DIR=/home/test_user02/lil/model/Roberta-large
SQUAD_DIR=/home/test_user02/lil/data/SQuAD/data

python3 evaluate.py \
  --rootdir=$BASE_DIR/20210917_095250 \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --idx_file=$BASE_DIR/idx.txt \
  --output_dir=$BASE_DIR \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --max_seq_length=512 \
  --version_2_with_negative=True

python3 $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $BASE_DIR/predictions.json \
  --na-prob-file $BASE_DIR/null_odds.json