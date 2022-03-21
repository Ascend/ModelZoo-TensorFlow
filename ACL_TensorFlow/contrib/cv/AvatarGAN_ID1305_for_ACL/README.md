# 模型功能

* 依据真实人脸照片生成对应卡通图像。
* 参考论文
  
  RoboCoDraw: Robotic Avatar Drawing with GAN-based Style Transfer and Time-efficient Path Optimization
[[PDF]](https://arxiv.org/abs/1912.05099)


# pb模型
```
python3.7  freeze_graph_entry.py 
```

# om模型
在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:
```bash
cd ${code_dir}/scripts/

sh model_convert.sh
```
推理使用的pb及om模型  [在此获取](https://pan.baidu.com/s/1w_CQyVib3omzeQt-B2Hniw )  （提取码：ep8n）
# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。


## 1.数据集转换bin
data_dir为测试集路径
data_bin_dir为转化后的的bin数据集路径

```
python3 jpg2bin.py --XtoY 'testA' --YtoX 'testB' \
            --datasets_dir '/home/ma-user/modelarts/inputs/data_url_0/avatar_data' \
            --output_dir '/home/ma-user/modelarts/outputs/train_url_0/bin'
```

## 2.推理

使用msame推理工具，发起推理测试，可参考 masame_inference.sh 

```bash
input_path_1=/输入avatar2real数据集路径
input_path_2=/输入real2avatar数据集路径
output_path_1=/avatar2real推理输出路径 
output_path_2=/real2avatar推理输出路径 

sh msame_inference.sh ${input_path_1} ${input_path_2} ${output_path_1} ${output_path_2}
```



## 3.推理结果后处理
推理结果为二进制.bin格式数据，需要将其转换为可视的.jpg格式图片。

```bash
input_dir_avatar2real=/输入avatar2real.bin数据路径
input_dir_real2avatar=/输入real2avatar.bin数据路径
output_dir_avatar2real=/avatar2real输出路径
output_dir_real2avatar=/real2avatar输出路径

python3.7 bin2jpg.py --in_XtoY ${input_dir_avatar2real} \
      --in_YtoX ${input_dir_real2avata} \
      --out_XtoY ${output_dir_avatar2real} \
      --out_YtoX ${output_dir_real2avatar}
```

## 4.推理样例
![pics](./img/inference_output_1.jpg)

