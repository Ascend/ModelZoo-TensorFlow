BERT_BASE_DIR=/home/test_user02/lil/model/Roberta-large
SQUAD_DIR=/home/test_user02/lil/data/SQuAD/data

python3 get_feature.py \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --max_seq_length=512 \
  --output_dir=/home/test_user02/lil/output/test \
  --version_2_with_negative=False