NETWORK='mobilenet'
DATASET='webface'
#STRATEGY='min_and_min'
STRATEGY='min_and_max'
#STRATEGY='hardest'
#STRATEGY='batch_random'
#STRATEGY='batch_all'
#MINE_METHOD='simi_online'
MINE_METHOD='online'
DATA_DIR='./dataset/casia-112x112'
PRETRAINED_MODEL="./train/models/facenet_ms_mp/model-20210916.ckpt-60000/model-20210916-145027.ckpt-60000"
#P=21
#K=10
P=41
K=5
#P=14
#K=15
#P=10
#K=21
#P=30
#K=7
NAME=${NETWORK}_${DATASET}_${STRATEGY}_${MINE_METHOD}_${P}_${K}
SAVE_DIR=E:/!UNet/MassFace-master/saved/triplet
LOCAL_CMD="python ./train/train_triplet.py --logs_base_dir ${SAVE_DIR}logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/${NAME}/  --image_size 224  --optimizer ADAGRAD --learning_rate 0.001 --weight_decay 1e-4 --max_nrof_epochs 10000  --network ${NETWORK} --dataset ${DATASET} --data_dir ${DATA_DIR} --pretrained_model ${PRETRAINED_MODEL} --random_crop --random_flip --image_size 112 --strategy ${STRATEGY} --mine_method ${MINE_METHOD} --num_gpus 1 --embedding_size 1024 --scale 10 --people_per_batch ${P} --images_per_person ${K}"
echo ${LOCAL_CMD} && eval ${LOCAL_CMD}
