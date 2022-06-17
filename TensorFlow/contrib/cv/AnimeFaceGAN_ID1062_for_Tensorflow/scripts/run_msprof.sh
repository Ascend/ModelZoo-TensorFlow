#!/bin/bash
set -e
### Before run this shell, make sure you have generated profiling data, and have installed CANN toolkit package
### refer to link: https://support.huaweicloud.com/Development-tg-cann202training1/atlasprofilingtrain_16_0015.html
### $1 is the absolute directory of profiling data.
### start commands sample: sh scripts/run_msprof.sh /home/npu_profiling

PROFILING_DIR=$1

## Be careful the $MSPROF_DIR, you may change it on different plateform
## arm architecture, `uname -a`
# MSPROF_DIR=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux/toolkit/tools/profiler/profiler_tool/analysis/msprof
## x86 architecture, `uname -a`     For Ai1S platform
MSPROF_DIR=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/toolkit/tools/profiler/profiler_tool/analysis/msprof

python3.7 ${MSPROF_DIR}/msprof.py import -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof sqlite.\n"

python3.7 ${MSPROF_DIR}/msprof.py query -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof query.\n"

python3.7 ${MSPROF_DIR}/msprof.py export timeline -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof timeline.\n"

python3.7 ${MSPROF_DIR}/msprof.py export summary -dir ${PROFILING_DIR}
echo "===>>>[OK] msprof summary.\n"