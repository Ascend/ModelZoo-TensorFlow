#! /bin/bash

DATA_DIR=/home/test_user03/tf_records/
MODEL_DIR=/home/test_user03/hh

#�����߸��˶���Ԥ�õ����ݼ���Ԥѵ��ģ�͡�ATC-OMģ�͵ȣ�֧�ִ�OBS������
#obsutil cp obs://obsxxx/xxx/xxx.ckpt ./model/ -f -r

#testcase���壬�����߸��ݲ�ͬģ��д��
python3 amoeba_net.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --num_cells=6 \
  --image_size=224 \
  --num_epochs=1 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --mode=predict \
  --iterations_per_loop=1251 \
  --num_train_images=10000 \
  --num_eval_images=1000 \
  --eval_timeout=10 \
  > predict.log 2>&1 
#���������������ֻ����1000��ͼƬ���������ӡ��Ϣ��predict.log

#����жϣ����ܼ�����ckpt/��־�ؼ��֡����ȼ��lossֵ/accucy�ؼ��֡����ܼ���ʱ���/ThroughOutput�ȹؼ���
key1="Restoring parameters from"  #���ܼ����
key2="Inference speed ="  #���ܼ����



if [ `grep -c "$key1" "predict.log"` -ne '0' ] && [ `grep -c "$key2" "predict.log"` -ne '0' ];then   #���Ը�����Ҫ��������߼�
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi