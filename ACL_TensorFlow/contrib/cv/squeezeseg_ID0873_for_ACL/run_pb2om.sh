#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc atc --model=/home/HwHiAiUser/AscendProjects/frozen_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/squeezeseg_acc --soc_version=Ascend310 --input_shape="Placeholder:1,64,512,5;Placeholder_1:1,64,512,1" --log=info --out_nodes="output:0"