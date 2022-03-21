export RANK_INDEX=0
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=2
export DEVICE_INDEX=0

export ENABLE_FORCE_V2_CONTROL=1
python3.7.5 train_on_npu.py --dataset=Market --data_path=./dataset/Market/ >train_on_npu.log 2>&1 &
