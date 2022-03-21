#!/bin/bash
set -e
### Before run this shell, make sure you have generated profiling data
### $1 is the absolute directory of profiling data 
### $MSPROF_DIR is only use for on Apulis Plateform, you may change it on other platform
PROFILING_DIR=$1
MSPROF_DIR=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux/toolkit/tools/profiler/profiler_tool/analysis/msprof

cd $MSPROF_DIR

python3.7 msprof.py import -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof sqlite.\n"

python3.7 msprof.py query -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof query.\n"

python3.7 msprof.py export timeline -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof timeline.\n"

python3.7 msprof.py export summary -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof summary.\n"