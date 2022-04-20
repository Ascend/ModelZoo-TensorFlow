#!/bin/bash

devices=$1
config_file=$2

source scripts/env.sh

function cmd() {
  device_id=$1
  device_rank=$2
  rank_id=$3
  config_file=$4
  rank_table_file=$5

  # Turn profiling on
  export JOB_ID=123456789
  export DEVICE_ID=${device_id}
  export DEVICE_INDEX=${device_id}
  export RANK_ID=${rank_id}
  export RANK_SIZE=${device_rank}
  if [ -n "$rank_table_file" ]; then
      export RANK_TABLE_FILE=${rank_table_file}
  fi

  export MOX_USE_NPU=1
  export FUSION_TENSOR_SIZE=2000000000
  export MOX_USE_TF_ESTIMATOR=0
  export MOX_USE_TDT=1

  export HEARTBEAT=1
  export CONITNUE_TRAIN=true
  export LOG_DIR=./log

  export ASCEND_GLOBAL_EVENT_LEVEL=0
  export ASCEND_GLOBAL_EVENT_ENABLE=0
  export ASCEND_GLOBAL_LOG_LEVEL=3
  export TF_CPP_MIN_LOG_LEVEL=3

  python3 src/main.py \
      --config-file ${config_file} \
      env.rank_size ${device_rank} \
      env.device 'npu'
}

# read device id to list
function mfcb { local val="$4"; "$1"; eval "$2[$3]=\$val;"; };
function val_ltrim { if [[ "$val" =~ ^[[:space:]]+ ]]; then val="${val:${#BASH_REMATH[0]}}"; fi; };
function val_rtrim { if [[ "$val" =~ [[:space:]]+$ ]]; then val="${val:0:${#val}-${#BASH_REMATH[0]}}"; fi; };
function val_trim { val_ltrim; val_rtrim; }

if [[ -z "$1" ]]; then
  echo "[INFO] device_id not set. Input argument could be like '1' or '0,1,2'."
  echo "[INFO] Set device_id=0 by default."
  device_list=0
  device_rank=1
else
  readarray -c1 -C 'mfcb val_trim device_list' -td, <<<"$devices,"; unset 'device_list[-1]'; declare -a device_list;
  device_rank=${#device_list[@]}
fi
echo "[INFO] device_list: ${device_list[@]}"
echo "[INFO] device_rank: ${device_rank}"

cur_dir=`pwd`
if [ $device_rank -gt 1 ]; then
  source_json=scripts/8p.json
  trimmed_dev_list=`echo ${device_list[@]} | tr -d ' '`
  if [ ${device_rank} -eq 8 ]; then
    target_json=$source_json
    echo "[INFO] 8p using source hccl config file: ${target_json} ..."
  else
    target_json=scripts/${device_rank}p_${trimmed_dev_list}.json
    echo "[INFO] (Re)Generating hccl config file: ${target_json} ..."
    python3 scripts/prepare_hccl_json.py ${devices} ${source_json} ${target_json}
  fi

  max_device_rank=`expr ${device_rank} - 1`
  for d_id in ${!device_list[@]}; do
    cd ${cur_dir}
    bash scripts/create_new_experiment.sh D_${device_list[$d_id]}
    cd D_${device_list[$d_id]}
    if [ $d_id -ne ${max_device_rank} ]; then
      cmd ${device_list[$d_id]} ${device_rank} ${d_id} ${config_file} ${target_json} &
    else
      cmd ${device_list[$d_id]} ${device_rank} ${d_id} ${config_file} ${target_json} && echo "[INFO] Train done."
    fi
  done
else
  cmd ${device_list[$d_id]} ${device_rank} 0 ${config_file}
fi
