#!/bin/bash

devices=$1
models=$2
dir=$3
codec_file=$4

if [ -z "$4" ]; then
  codec_file=configs/codecs/default_sdr_x264.json
fi
video_file_ext=$(cat ${codec_file} | python3 -c "import sys, json; print(json.load(sys.stdin)['format'])")

# Set models
declare -A edvr_s2=(
["config"]="cmodels/2x/edvr_modified_scale2_config.py"
["ckpt"]="models/2x/EDVR-450000"
)
declare -A edvr_s4=(
["config"]="cmodels/4x/edvr_modified_scale4_config.py"
["ckpt"]="models/4x/EDVR-600000"
)

readarray -d , -t models <<< "${models},"
unset 'models[$((${#models[@]}-1))]'

# Remove the last / if it exists
echo "$dir" | grep '/$'
if [ $? -eq 0 ]
then
  dir=${dir%/}
fi


# Create temp txt file to record subvideo names
cur_dir=`pwd`

source scripts/env.sh

function cmd() {
  device_id=$1
  device_rank=$2
  rank_id=$3
  model_name=$4
  dir_in=$5
  dir_out=$6
  io_backend=$7

  # Turn profiling on
  export JOB_ID=123456789
  export DEVICE_ID=${device_id}
  export DEVICE_INDEX=${device_id}
  export RANK_ID=${rank_id}
  export RANK_SIZE=${device_rank}

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

  rm -rf kernel_meta
  rm -rf ~/ascend/log/plog

  declare -n model="$model_name" # model is a reference
  python3 src/main.py \
    --config-file ${model["config"]} \
    mode inference \
    data.data_dir ${dir_in} \
    data.inference.auto_adapt_input True\
    inference.result_dir ${dir_out} \
    inference.io_backend 'disk' \
    env.rank_size ${RANK_SIZE} \
    checkpoint ${model["ckpt"]} \
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


cnt=0
io_backend="disk"
for model_name in "${models[@]}"; do
  cnt=$(( $cnt + 1 ))

  dir_out="${dir}_${model_name}"
  if [ ! -d ${dir_out} ]; then
    mkdir ${dir_out}
  fi

  # set video output for the last model
  if [ $cnt -eq ${#models[@]} ]; then
    io_backend="disk:ffmpeg"
    if [ ! -d "${dir_out}_videos" ]; then
    mkdir ${dir_out}_videos
  fi
  fi

  if [ $device_rank -gt 1 ]; then
    max_device_rank=`expr ${device_rank} - 1`
    for d_id in ${!device_list[@]}; do
      cd ${cur_dir}
      bash scripts/create_new_experiment.sh D_${d_id}
      cd D_${d_id}

      # inference
      if [ $d_id -ne ${max_device_rank} ]; then
        cmd ${device_list[$d_id]} ${device_rank} ${d_id} ${model_name} ${dir} ${dir_out} ${io_backend} &
      else
        cmd ${device_list[$d_id]} ${device_rank} ${d_id} ${model_name} ${dir} ${dir_out} ${io_backend} || exit 1
      fi

    done
    # wait untill all jobs done
    wait < <(jobs -p)
  else
    cmd ${device_list[$d_id]} ${device_rank} ${device_list[$d_id]} ${model_name} ${dir} ${dir_out} ${io_backend} || exit 1
  fi
  # update path
  dir="${dir_out}"
done

