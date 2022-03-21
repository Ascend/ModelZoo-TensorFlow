#!/bin/bash


cur_path=`pwd`/../


if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
cd $cur_path
start=$(date +%s)

nohup python3 run_classifier.py   --task_name=lcqmc_pair   --do_train=true   --do_eval=false   --data_dir=$TEXT_DIR   --vocab_file=./albert_config/vocab.txt  \
    --bert_config_file=./albert_config/albert_config_tiny.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4  --num_train_epochs=5 \
    --output_dir=./albert_lcqmc_checkpoints --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

step_sec=`grep -a 'INFO:tensorflow:global_step/sec' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk 'END {print $2}'`
Performance=`echo "scale=2; 1000/$step_sec" | bc`
#echo "Final Precision MAP : $average_prec"
echo "Final Performance ms/step : $Performance"
echo "Final Training Duration sec : $e2etime"  



