# monodepth
## 原始项目链接

https://github.com/mrharicot/monodepth



## 精度测试结果

|                     | abs_rel | sq_rel | rmse  | rmse_log | d1-all | a1    | a2    | a3    |
| ------------------- | ------- | ------ | ----- | -------- | ------ | ----- | ----- | ----- |
| 论文                | 0.124   | 1.388  | 6.125 | 0.217    | 30.272 | 0.841 | 0.936 | 0.975 |
| 基线（改写dataset） | 0.1217  | 1.3121 | 6.162 | 0.217    | 30.987 | 0.839 | 0.934 | 0.973 |
| NPU910              | 0.1215  | 1.3363 | 6.135 | 0.215    | 30.796 | 0.841 | 0.938 | 0.975 |



## 代码和数据集

**本次迁移选择KITTI数据集**

> 【特殊说明】按照github源码方式下载数据集比较困难，可以直接到obs桶里面获取已经处理好的数据集。

获取数据集：obs://cann-id2099/dataset/KITTI/



## 快速开始

1. 目录说明

   > 数据集目录 /home/disk/xjk/dataset/KITTI/
   >
   > 代码目录 ~/xjk/monodepth

    注意：

   - 运行命令时需要切换到monodepth目录下
   - --data_path 表示数据集路径
   - -- filename_file 表示训练和测试图片路径

2. 训练

   ```shell
   python3.7 ./monodepth_main.py --mode train --model_name my_model --filenames_file ./utils/filenames/kitti_train_files.txt --data_path /home/disk/xjk/dataset/KITTI/ --log_directory ~/tmp/
   ```

3. 测试

   ```shell
   python3.7 monodepth_main.py --mode test --data_path /home/disk/xjk/dataset/KITTI/ --filenames_file ./utils/filenames/kitti_stereo_2015_test_files.txt --log_directory ~/tmp/ --checkpoint_path ~/tmp/my_model/model-181250
   ```

4. 获取精度

   ```shell
   python3.7 ./utils/evaluate_kitti.py --split kitti --predicted_disp_path ~/tmp/my_model/disparities.npy --gt_path /home/disk/xjk/dataset/KITTI/
   ```
