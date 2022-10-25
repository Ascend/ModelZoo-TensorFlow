数据集链接为 https://storage.googleapis.com/keypose-transparent-object-dataset/models.zip

运行命令为python3 -m keypose.trainer configs/bottle_0_t5 /tmp/model 其中configs的default_base.yaml的step设置为70,000

运行最终几步结果为： 69995 Keypose loss: 0.00380851841 0.000805588672 0.00300292973 0 1 [115.585976 56.4450188 1] [352.585968 139.554977 67.5676651] [352.345062 140.143356 67.4446564] 69996 Keypose loss: 0.0045566014 0.000759354443 0.00379724707 0 1 [107.608925 60.8226662 1] [378.608917 204.177338 77.6202087] [378.270599 203.274796 77.3878632] 69997 Keypose loss: 0.00427989196 0.000648553 0.00363133894 0 1 [120.93557 22.0719337 1] [438.935577 126.07193 83.3570633] [437.187561 124.521378 84.5888596] 69998 Keypose loss: 0.00512055028 0.00100593979 0.00411461061 0 1 [137.617661 46.038475 1] [298.617676 181.038483 58.1664925] [300.742401 180.244308 58.3245926] 69999 Keypose loss: 0.00624349574 0.000914623786 0.00532887224 0 1 [118.494125 38.7468567 1] [518.494141 176.746857 87.2072372] [519.505249 178.638626 87.5264816]

最终loss曲线与keypose提供的bottle_0_t5的log文件误差不超过0.5%，精度达标。

离线推理： 首先利用ckpt2pb.py文件，对keypose模型中生成的ckpt文件进行转化，得到对应的pb文件。

然后再使用命令行： atc --model=./frozen_model.pb --framework=3 --output=./om_model --soc_version=Ascend310 --input_shape="img_L:1,120,180,3;img_R:1,120,180,3;offsets:1,3;hom:1,3,3" --log=info 即可得到对应的om文件

操作示例： 我们这里采用1711step所保存的ckpt文件进行离线推理，即model.ckpt-1711(.meta/.index/.data-00000-of-00001) 通过python3 ckpt2pb.py，得到frozen_1711.pb，网盘链接为：https://pan.baidu.com/s/1pOQgxlHiMcgneqa1cHvOuw?pwd=xe0u （提取码：xe0u） 然后再运行上述atc命令，得到对应的pb_om_model.om，网盘链接为链接:https://pan.baidu.com/s/1GjbCvPzwjCu6qcrmDCxS_w?pwd=7gcz (提取码：7gcz)

验证推理： 参考https://gitee.com/ascend/tools/tree/master/msame， 我们使用msame推理工具，进行推理测试，命令为： ./msame --model "/home/pb_om_model.om" --input "/home/keypose/data/bottle_0/texture_5_pose_0/" --output "/home/keypose/" --outfmt TXT

生成的结果为txt格式，对比NPU训练的结果，数值完全一致，推理成功。