cd ../

:<<!
--data_dir 数据集中测试集路径
--data_mode 测试集中的scale，可选[2x,4x,8x],对应于要离线推理的模型的输入数据
--dst_path  转换的bin文件的存储位置
--pic_num 转换的图片数量，当为-1时，全部转换
--batch_size 转换为bin文件时，每个bin文件包含的图片个数，其应当与atc模型转换时--input_shape参数的bz相同
!
python img_preprocess.py --data_dir /mnt/data/wind/dataset/SRDRM/USR248/TEST/ \
                         --data_mode 8x \
                         --dst_path ./input \
                         --pic_num -1 \
                         --batch_size 2