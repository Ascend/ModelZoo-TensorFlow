cd code
pip3 install -r requirements.txt
mkdir TFRecord_food101
mkdir temp
echo "prepare success!"

# ./obsutil config -i=H2I0TJ6MWJQ7OCYWEX1C -k=g33WG4Y7CRRF9zYaOqpMqHDrqxqqU5XBizP8NcOC -e=112.95.163.82
./obsutil cp obs://sunshk/TFRecord_food101/train-00000-of-00128 TFRecord_food101/train-00000-of-00128
./obsutil cp obs://sunshk/TFRecord_food101/validation-00000-of-00016 TFRecord_food101/validation-00000-of-00016
./obsutil cp obs://sunshk/food_ckpt.tar food_ckpt.tar

echo "obsutil success!"

tar zxvf food_ckpt.tar

echo "tar success!"

DATA_DIR=./TFRecord_food101
MODEL_DIR=./temp
PRETRAINED_PATH=./food_ckpt


python main_classification.py \
--dataset_name=food101 \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--pretrained_model_checkpoint_path=${PRETRAINED_PATH} \
--num_gpus=1 \
--mixup_type=1 \
--autoaugment_type=good \
--label_smoothing=0.1 \
--resnet_version=2 \
--resnet_size=50 \
--anti_alias_filter_size=3 \
--anti_alias_type=sconv \
--use_sk_block=True \
--use_dropblock=True \
--dropblock_kp="0.9,0.7" \
--batch_size=64 \
--preprocessing_type=imagenet_224_256a \
--base_learning_rate=0.001 \
--learning_rate_decay_type=cosine \
--lr_warmup_epochs=5 \
--train_epochs=1 \
--bn_momentum=0.966 \
--weight_decay=0.00001 \
--keep_checkpoint_max=0 \
--ratio_fine_eval=1.0 \
--epochs_between_evals=1 \
--clean
--train_regex=train-00000-of-00128 \
--val_regex=validation-00000-of-00016 > train.log 2>&1


if [ `grep -c "Benchmark" "train.log"` -ne '0' ] ;then
	echo "Run testcase success!"
else
	echo "Run testcase failed!"