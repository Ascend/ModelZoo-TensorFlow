start_time=`date +%s`
python3 python tprnn_train_human.py -type LSTM -epoch 1 -lr 0.005 -batch_size 16 > train.log
end_time=`date +%s`

#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="Minibatch Loss"  #功能检查字1

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.