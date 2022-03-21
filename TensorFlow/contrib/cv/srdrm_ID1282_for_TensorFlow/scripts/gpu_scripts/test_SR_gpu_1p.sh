cd ../../

:<<!
以下命令执行推理(可以修改的参数如下所示)
--test_mode 可选[2x,4x,8x] 执行训练所用训练数据集
--data_dir 测试数据集的路径
--model_name 使用的模型
--test_epoch 测试使用的模型为test_epoch保存的模型,srdrm-gan 选 64, srdrm 选 52
!

python test_SR.py --test_mode 8x \
                        --chip gpu \
                        --data_dir /mnt/data/wind/dataset/SRDRM/USR248/TEST/ \
                        --model_name srdrm-gan \
                        --test_epoch 52 \
                        --platform linux