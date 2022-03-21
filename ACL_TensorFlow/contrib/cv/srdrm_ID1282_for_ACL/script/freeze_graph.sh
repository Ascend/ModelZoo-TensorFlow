cd ../

:<<!
以下命令执行推理(可以修改的参数如下所示)
--input_model 模型的权重文件.h5
--input_model_json 模型的网络结构文件,json
--output_model 输出的pb文件的保存路径
!
python keras_to_tensorflow.py  \
       --input_model="/mnt/data/wind/SRDRM/checkpoints/USR_8x/srdrm/model_60_.h5" \
       --input_model_json="/mnt/data/wind/SRDRM/checkpoints/USR_8x/srdrm/model_60_.json"  \
       --output_model="/mnt/data/wind/SRDRM/checkpoints/freeze_model/srdrm/model_52_g.pb"