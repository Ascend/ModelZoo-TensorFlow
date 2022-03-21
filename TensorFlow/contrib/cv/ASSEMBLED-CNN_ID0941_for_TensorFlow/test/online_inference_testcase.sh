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
--anti_alias_filter_size=3 \
--val_regex=validation-00000-of-00016 > train.log 2>&1


if [ `grep -c "Benchmark" "train.log"` -ne '0' ] ;then
	echo "Run testcase success!"
else
	echo "Run testcase failed!"


