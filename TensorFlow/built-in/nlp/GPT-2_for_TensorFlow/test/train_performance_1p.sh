#!/bin/bash


cur_path=`pwd`/../
cd $cur_path

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
cd $cur_path/gpt_2_simple
start=$(date +%s)

nohup python3 -u train.py 100 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

full_sec=`grep -a 'loss=' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| tail -1| awk -F "[[,], ]" '{print $4}'`
total_step=`grep -a 'loss=' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| tail -1| awk -F "[[,], ]" '{print $2}'`
step_sec=`echo "scale=2; ${full_sec}*1000/${total_step}" | bc`


#Performance=`awk 'BEGIN{printf "%.2f\n",'1000'/$step_sec}'`
#echo "Final Precision MAP : $average_prec"
echo "Final Performance ms/step : $step_sec"
echo "Final Training Duration sec : $e2etime"  



