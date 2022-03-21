DATA_DIR=/home/sunshk/assemble/data/TFRecord_food101
MODEL_DIR=/home/sunshk/food_ckpt

ASCEND_DEVICE_ID=0 python main_classification.py \
--eval_only=True \
--dataset_name=food101 \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--preprocessing_type=imagenet_224_256a \
--resnet_version=2 \
--resnet_size=50 \
--use_sk_block=True \
--use_resnet_d=False \
--anti_alias_type=sconv \
--anti_alias_filter_size=3 
