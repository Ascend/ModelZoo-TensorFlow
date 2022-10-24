#### SMITH (Siamese Multi-depth Transformer-based Hierarchical Encoder) Model Code

## 训练环境
* TensorFlow 1.15.0
* Python 3.7.0

## 迁移部分代码[地址](https://gitee.com/ascend/ModelZoo-TensorFlow/pulls/790)

## 代码及路径解释
```
smith_ID2025_for_ACL
├── ckpt2pb.py              ckpt模型固化为pb
├── ckpt2pb.sh              ckpt模型固化为pb脚本
├── atc.sh  				act工具 pb==>om 转换命令
├── msame.sh				msame工具：om离线推理命令
├── gen_bin_by_img2bin.sh   推理数据预处理：将txt编码数据文件转换为bin并进行预处理
├── data4input				存放输入到img2bin工具前的txt文本(数据路径见README.md)		
│   └── ..			        样例数据
├── data4output		        img2bin转换后bin文件位置
│   └── ..			            
├── msame_output		    msame推理结果文件存放位置		
    └── ..			        
```

## 离线推理
### 1、原始模型转PB模型
#### ckpt转pb
```
python3 ckpt2pb.py --dual_encoder_config_file=smith/config/dual_encoder_config.smith_wsp.32.48.pbtxt
```
#### 参数说明
```
--dual_encoder_config_file  模型依赖的smith相关配置文件
--ckpt_path  ckpt的路径文件
--output_graph  输出的节点名称
```
*注*：ckpt转pb的时候依赖smith原始的代码。 pb模型下载路径

### 2、PB模型转OM模型
使用ATC模型转换工具转换PB模型到OM模型
```
atc --model=smith.pb --framework=3 --output=pb_res --soc_version=Ascend910 --input_shape="input_ids_1:32,2048;input_mask_1:32,2048;input_ids_2:32,2048;input_mask_2:32,2048" --out_nodes="seq_rep_from_bert_doc_dense/l2_normalize_1:0;Sigmoid:0;Round:0" --log=debug
```
[下载pb模型](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OVWrEUKDR1ADCmJjkBGQ3htGaizfvxlSf9IlnYwV2/viY4ioin+PR2KRtwadLKxE1gt4UUhppUs5f/woysQbGg9EGEcwk+17FznTTlVaDwMq/+IOPf44FDjQSSDcfB80gaA9iw1wn0I54iM5Ay3J2PDUDpr9bDe3faMdnnFv05ROdjODdyWRuoJPBK4YRNwwGAEJ7qhjWbF0IbVkmzXKPP8nO8WnY+e9O6RA5Jl6wlGkkz5Yx+o7b6GD3LNbjhYdw6z5rXkcROydKeC3wsEwwwqbVhuJG6kLgL9fqFksBp+0JwC6Y5C3EcliWoQqKTW+CAXCwjkDybKTKHubH/sxc/uGet+jg85rV9ZGEhdCoK07K8T6O2KxWSq746MYu38l8Q/Py7TZFTP7tbrLqXZUcm1olvl7Iq+smaLS5Jt9ZlQdO9pndkSiy95idXN6IzGblo0j7GxZlNokjnqAyBknbgu8R92C8jDEW9liJzokLrHYLKF24DiM/I1UlTJxbzrXBOSgng1b+Lb+0DMsWkqKyQ==)
密码：111111

### 3、OM的输入数据准备（使用img2bin工具将编码转成bin）
参考[img2bin](https://gitee.com/ascend/tools/tree/master/img2bin)
```
python3 img2bin.py -i ./input_ids_1.txt -t int32 -o ./out/
python3 img2bin.py -i ./input_ids_2.txt -t int32 -o ./out/
python3 img2bin.py -i ./input_mask_1.txt -t int32 -o ./out/
python3 img2bin.py -i ./input_mask_2.txt -t int32 -o ./out/
```
所需要的测试txt数据见：[data](https://gitee.com/ascend/tools/tree/master/img2bin)

输入的bin文件
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OVWrEUKDR1ADCmJjkBGQ3htGaizfvxlSf9IlnYwV2/viY4ioin+PR2KRtwadLKxE1gt4UUhppUs5f/woysQbGg9EGEcwk+17FznTTlVaDwMq/+IOPf44FDjQSSDcfB80gaA9iw1wn0I54iM5Ay3J2PDUDpr9bDe3faMdnnFv05ROdjODdyWRuoJPBK4YRNwwGAEJ7qhjWbF0IbVkmzXKPDE+ychH6CD2Ka1+370tOv+EstWuSZKPSGUmnci/52B9QiRiqpqcji3pWX79RAfSjDT1yVCPHWkJteShopJrqhOD95uxfvfBkXMMe9JxygtLAErbO5DTKjsVVZcexX4h6mEIs5qHwQ8I3jeEv8jf37JPhxBdTFs8KMJFQeP5bP9BW8QhboApxuECPvBtELJYOY4tSM4ykQzX3TbKjw3in0sQozx0Vx86Q7zT/xNNPUQ/Qrh59Dox6w/kXlISs1uCYH83Ou7wino3gilFsV6xJY0OiWdxVhBevg3wKS+actgxQV6+GEOpKsDc/jwozfs8Ps+tgXtxIaopxpLijbLgMNo=)
密码：111111

### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/master/msame)

### 5、om模型推理精度测试
使用如下命令进行精度测试
```
./msame --model "pb_res.om" --input "out/out4tmp/input_ids_1.bin,out/out4tmp/input_ids_2.bin,out/out4tmp/input_mask_1.bin,out/out4tmp/input_mask_2.bin" --output "output" --loop 1 --outfmt TXT --debug true
```
[下载OM模型](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OVWrEUKDR1ADCmJjkBGQ3htGaizfvxlSf9IlnYwV2/viY4ioin+PR2KRtwadLKxE1gt4UUhppUs5f/woysQbGg9EGEcwk+17FznTTlVaDwMq/+IOPf44FDjQSSDcfB80gaA9iw1wn0I54iM5Ay3J2PDUDpr9bDe3faMdnnFv05ROdjODdyWRuoJPBK4YRNwwGAEJ7qhjWbF0IbVkmzXKPLD4mddW+FqOmTHO+HQQ1dkVprrpmld9fBPkjzKd90AF7HK7xZmyJXMoVmrge8lYqO52NksQAKJcnLqPro0N7ZFSkQOk0Nx/IxQtCbSeQXfqcdnptyvegb/KrGK6sDulP1Ys4AwoZ7h152uqEDwwdvkNCBHmrf1ctMHGcRI0Fo0JS+g5iUnVVJAXHAlrlmyIoxJRKCkRUyfOa6xOZrAWwKcDwsTs+u1xwTV3yKZZ271RoK4DaJEnilvvEzPZL8MG/6qlQcvTvgrUitfWsfLvLODVDGV3dlcxQIgCZSz05Mm88RGVc4GF0Aarkc21pdir8A==)
密码：111111

[msame推理结果](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OVWrEUKDR1ADCmJjkBGQ3htGaizfvxlSf9IlnYwV2/viY4ioin+PR2KRtwadLKxE1gt4UUhppUs5f/woysQbGg9EGEcwk+17FznTTlVaDwMq/+IOPf44FDjQSSDcfB80gaA9iw1wn0I54iM5Ay3J2PDUDpr9bDe3faMdnnFv05ROdjODdyWRuoJPBK4YRNwwCXEASKWlmsbC2Bd/rv+QuNYGdPjPrRIaI+d1EZyQ4n/2l15OOQepYvOewkIMELc0ztuwDR2CNlFoQc9c/Hk7rv9wdJoU8AP7H6MjAL7VnOLRvdHqRo3u6tJDubb40M9/w0FOXpaPVRlYB5velWBcEvlvgQxbKXWyCrvS91+1B5ZHxy0EBfe1wBwi+dhkIZHpfWTDSux2zE6O/USQEWxbUuLke7ile9KhC+M/MAZfHATQ051oyPvnK74hUnCyuBe2BjePna6Wyrra5qlAVEvFtrWrBCrZzLPMNBClImnwY6PZlUvJFm4unwZ4N/kH6a41/yQXU5yGP+UkRJNX4Te6VkvkMhNs6yY0A/HhjurP9CA=)
密码：111111

### 6 精度对比
##### 用OM推理的结果：
```
得分结果：
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.99707 0.5 0.99707 0.99707 0.5 0.99707 0.99707 0.5 0.5 0.5 0.5 0.5 0.5 0.99707 0.5 0.5 0.5 0.99707 0.5 0.5 0.5 0.99707 0.99707
对应label结果：
0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1
```

##### 用ckpt直接推理的结果：
```
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.99751484", "predicted_class":"1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.99752337", "predicted_class":"1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.9975246", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.99752116", "predicted_class": "1.0"}
{"predicted_score": "0.9975234", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.99751836", "predicted_class": "1.0"}
{"predicted_score": "0.9975234", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.99752474", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.9975234", "predicted_class": "1.0"}
{"predicted_score": "0.9975238", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.9975239", "predicted_class": "1.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.5", "predicted_class": "0.0"}
{"predicted_score": "0.9975251", "predicted_class": "1.0"}
{"predicted_score": "0.99752605", "predicted_class": "1.0"}
```
