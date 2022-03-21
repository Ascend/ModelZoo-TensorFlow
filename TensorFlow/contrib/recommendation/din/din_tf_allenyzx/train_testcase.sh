set -x
export POD_NAME=another0

execpath=${PWD}

output='./output'
rm -rf *.pbtxt
ulimit -c 0

start_time=`date +%s`

python3  npu_train.py \
--embed_dim=8 \
--maxlen=40 \
--ffn_activation=prelu \
--learning_rate=0.001 \
--batch_size=512 \
--dnn_dropout=0 \
--max_steps=19000 \
--save_checkpoint_steps=19000 >train.log 2>&1

end_time=`date +%s`
#结果判断，功能检查输出ckpt/日志关键字、精度检查loss值/accucy关键字、性能检查耗时打点/ThroughOutput等关键字
key1="[GEOP]"  #功能检查字
key2="xxx"  #性能检查字
key3="xxx"  #精度检查字

if [ `grep -c "$key1" "train.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

echo execution time was `expr $end_time - $start_time` s.
