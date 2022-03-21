BookCorpus_DIR=/home/TestUser03/bupt_lil/Data/BookCorpus
BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large

python3 create_pretraining_data.py \
  --input_file=$BookCorpus_DIR/test_bookscorpus.txt \
  --output_file=$BookCorpus_DIR/pretrain_data \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5