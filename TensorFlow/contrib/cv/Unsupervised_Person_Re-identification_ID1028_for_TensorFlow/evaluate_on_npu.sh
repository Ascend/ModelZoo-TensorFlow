export RANK_INDEX=0
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=2
export DEVICE_INDEX=0

python3.7.5 evaluate_on_npu.py --dataset=Market >evaluate_on_npu.log 2>&1 &
