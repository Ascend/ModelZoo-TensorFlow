cd code

DATA_DIR=./TFRecord_food101
MODEL_DIR=./temp
PRETRAINED_PATH=./food_ckpt


python3  main_classification.py \
--eval_only=True \
--dataset_name=food101 \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--pretrained_model_checkpoint_path=${PRETRAINED_PATH} \
--resnet_version=2 \
--resnet_size=50 \
--use_sk_block=True \
--use_resnet_d=False \
--anti_alias_type=sconv \
--train_regex=train-00000-of-00128 \
--val_regex=validation-00000-of-00016 


