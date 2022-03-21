set -x
#export POD_NAME=another0

export install_path=/usr/local/Ascend/nnae/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/add-ons/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:${install_path}/fwkacllib/python/site-packages/hccl:$PYTHONPATH
export ASCEND_OPP_PATH=/usr/local/Ascend/nnae/latest/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux

export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=200


export RANK_TABLE_FILE=./hccl_8p.json
export RANK_INDEX=0

export JOB_ID=10086
export PRINT_MODEL=1
export RANK_SIZE=8

ulimit -c 0

#execpath=${PWD}
/usr/local/Ascend/toolkit/bin/adc --host 127.0.0.1:22118 --log "SetLogLevel(0)[info]"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu

# python3.7  train.py 
for((i=0;i<8;i++)); 
do
 export DEVICE_ID=$i
 export RANK_ID=$i
 python3.7 -u npu_8_distribute_train.py &
done

~  
