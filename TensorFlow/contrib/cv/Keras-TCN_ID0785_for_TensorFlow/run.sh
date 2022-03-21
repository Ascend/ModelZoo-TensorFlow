#!/bin/bash

echo "`date +%Y%m%d%H%M%S`-[INFO] start to run train_1p_ci.sh "

python3 /home/work/user-job-dir/mytask/train.py \
        --epoch=1 \
        --epochs=50 \
        --train_data_n=200000 \
        --validation_data_n=40000 \
        --seq_length=600 \
        --dropout_rate=0.1 \
        --lr=0.005 \
        --tensorboard_path='/cache/tfboard1' \
        --h5_obs_path='obs://kyq/mytestnew/mytestnew1h5/tcn.h5' \
        --tensorboard_obs_path='obs://kyq/mytestnew/tfboard1'

python3 -c "import moxing as mox;mox.file.copy('obs://kyq/mytestnew/log/job8f159cf5-job-mytestnew-0.log', '/cache/job8f159cf5-job-mytestnew-0.log');"

#从日志中提取loss和性能信息
awk -v epochs="50" 'BEGIN{count=0;print"BEGIN";}
$0~"val_loss" {count++;print "Epoch " count "/"epochs"\n" $0;}
END{print"END";}' /cache/job8f159cf5-job-mytestnew-0.log >> /cache/loss+perf_npu.txt

python3 -c "import moxing as mox;mox.file.copy('/cache/loss+perf_npu.txt', 'obs://kyq/loss+perf_npu.txt')"

echo "`date +%Y%m%d%H%M%S`-[INFO] finish to run train_1p_ci.sh "