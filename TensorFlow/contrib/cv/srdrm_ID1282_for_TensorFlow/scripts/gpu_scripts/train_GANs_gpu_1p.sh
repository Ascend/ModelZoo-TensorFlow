cd ../../

:<<!
以下命令执行训练srdrm-gan模型(可以修改的参数如下所示)
--train_mode 可选[2x,4x,8x] 执行训练所用训练数据集
--num_epochs 模型训练的总epoch
--ckpt_interval 模型每隔多少epoch进行保存
--sample_interval 模型每隔多少step进行生成结果的采样输出
--batch_size
--data_path 数据集的路径
--start_epoch 从第几个epoch开始训练，默认从头开始
!

python train_GANs.py --train_mode 8x \
                           --chip gpu \
                           --num_epochs 70 \
                           --ckpt_interval 4 \
                           --sample_interval 500 \
                           --model_name srdrm-gan \
                           --batch_size 2 \
                           --data_path /mnt/data/wind/dataset/SRDRM/USR248/ \
                           --start_epoch 0 \
                           --platform linux
