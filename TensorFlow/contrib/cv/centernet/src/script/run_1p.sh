#!/bin/bash
LOG_PREFIX="<run_1p>"

cd "$(dirname "$0")" || exit
SOURCE_DIR=$(dirname "$(pwd)")
WORKSPACE_DIR=$(dirname "${SOURCE_DIR}")
cd "$WORKSPACE_DIR" || exit
export PRJPATH=${WORKSPACE_DIR}
echo "${LOG_PREFIX}WORKSPACE_DIR = ${WORKSPACE_DIR}"
echo "${LOG_PREFIX}SOURCE_DIR = ${SOURCE_DIR}"

DATASET="VOC"
DATASET_URL=$2
echo "${LOG_PREFIX}DATASET_URL = ${DATASET_URL}"
echo "${LOG_PREFIX}=== Download dataset ==="
python "${SOURCE_DIR}/script/mox_copy.py" "$DATASET_URL" "./dataset/$DATASET/tfrecord/"
echo "${LOG_PREFIX}List dataset"
ls "$WORKSPACE_DIR/dataset/$DATASET/tfrecord/"

cd "$SOURCE_DIR" || exit
echo "${LOG_PREFIX}=== Load envs for CANN ==="
source ./script/env.sh

echo "${LOG_PREFIX}=== Run ==="
python ./test/test_centernet.py --platform config.platform.cfg_ascend

echo "${LOG_PREFIX}=== Upload log ==="
python "${SOURCE_DIR}/script/mox_copy.py" "/cache/dump_data" "./dataset/$DATASET/tfrecord/"
