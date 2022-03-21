export JOB_ID=10086
export ASCEND_DEVICE_ID=0

#rm -rf output

python3 freeze_graph.py \
  --bert_config_file=/home/test_user02/lil/model/Roberta-large/config.json \
  --output_dir=/home/test_user02/lil/output/msame/squadv2 \
  --ckpt_dir=/home/test_user02/lil/data/ckpt/squadv1/model.ckpt-43800 \
  --max_seq_length=512