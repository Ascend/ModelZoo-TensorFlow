#!/bin/bash
lidar_path="./bin_out/lidar"
lidar_mask_path="./bin_out/lidar_mask"
pred_cls_path="./bin_out/pred_cls"
ulimit -c 0
$HOME/AscendProjects/tools/msame/out/msame --model squeezeseg_acc.om --input ${lidar_path},${lidar_mask_path} --output ${pred_cls_path}



