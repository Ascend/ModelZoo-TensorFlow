#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc atc --model=/root/randlanet_final_version.pb --framework=3 --output=/root/randlanet --soc_version=Ascend310 --input_shape="xyz_0:3,40960,3;xyz_1:3,10240,3;xyz_2:3,2560,3;xyz_3:3,640,3;xyz_4:3,160,3;neigh_idx_0:3,40960,16;neigh_idx_1:3,10240,16;neigh_idx_2:3,2560,16;neigh_idx_3:3,640,16;neigh_idx_4:3,160,16;sub_idx_0:3,10240,16;sub_idx_1:3,2560,16;sub_idx_2:3,640,16;sub_idx_3:3,160,16;sub_idx_4:3,80,16;interp_idx_0:3,40960,1;interp_idx_1:3,10240,1;interp_idx_2:3,2560,1;interp_idx_3:3,640,1;interp_idx_4:3,160,1;rgb:3,40960,6" --log=info --out_nodes="probs:0"