
:<<!
以下命令执行推理(可以修改的参数如下所示)
--data_input 离线推理结果bin文件存放目录
--dst_path 离线推理结果bin文件转换为jpg文件的存放目录
--gt_dir ground_truth data
--batch_size 请于之前的步骤的bz保持相同
!

cd ../
python bin2img_measure.py  \
       --data_input /mnt/data/wind/SRDRM/offline_infer/output/20211024_14_26_9_817613 \
       --dst_path ./val2 \
       --gt_dir /mnt/data/wind/dataset/SRDRM/USR248/TEST/hr \
       --batch_size 2