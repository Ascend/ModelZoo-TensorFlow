export TF_CPP_MIN_LOG_LEVEL=2
python3.7 train.py \
--input_size=512 \
--learning_rate=0.0001 \
--batch_size_per_gpu=14 \
--num_readers=16  \
--max_steps=150000  \
--checkpoint_path=./checkpoint/ \
--training_data_path=./ocr/icdar2015/ \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt