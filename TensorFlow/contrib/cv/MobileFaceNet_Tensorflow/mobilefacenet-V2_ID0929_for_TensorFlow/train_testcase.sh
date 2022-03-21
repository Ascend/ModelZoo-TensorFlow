start_time=`date +%s`

python3 ./MobileFaceNet_Tensorflow/train_nets.py  \
--eval_db_path=./MobileFaceNet_Tensorflow/datasets/faces_ms1m_112x112 \
--tfrecords_file_path=./MobileFaceNet_Tensorflow/datasets/tfrecords \
--summary_path=./MobileFaceNet_Tensorflow/output/summary \
--ckpt_path=./MobileFaceNet_Tensorflow/output/ckpt \
--ckpt_best_path=./MobileFaceNet_Tensorflow/output/ckpt_best \
--log_file_path=./MobileFaceNet_Tensorflow/output/logs \
--arch_text=./MobileFaceNet_Tensorflow/arch/txt/MobileFaceNet_Arch.txt \
--var_text=./MobileFaceNet_Tensorflow/arch/txt/trainable_var.txt \
--show_info_interval=10 \
--validate_interval=100 \
--less_steps=150 > train.log
end_time=`date +%s`

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="total loss"  #功能检查字1

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then  #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.