export TF_CPP_MIN_LOG_LEVEL=3
export RANK_SIZE=1

python3 npu_train.py \
--embed_dim=8 \
--maxlen=40 \
--ffn_activation=prelu \
--learning_rate=0.001 \
--batch_size=512 \
--dnn_dropout=0 \
--max_steps=19000 \
--save_checkpoint_steps=19000