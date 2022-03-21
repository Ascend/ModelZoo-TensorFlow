# 模型功能

* 循环生成对抗网络，实现图像域到图像域之间的翻译。
* Original paper: https://arxiv.org/abs/1703.10593


# pb模型
```
cd ${code_dir}/offlie_inference/
python3.7  freeze_graph.py \
          --checkpoint_dir=${checkpoint_dir} \
          --XtoY_model=horse2zebra.pb \
          --YtoX_model=zebra2horse.pb \
          --image_size=256

```

# om模型
在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:
```bash
cd ${code_dir}/scripts/

sh model_convert.sh
```
推理使用的pb及om模型  [在此获取](https://pan.baidu.com/s/19PoDb9UbI4d_iXMu5duQvA)  （提取码：ejzz）

# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。


## 1.数据集转换bin
data_dir为测试集路径
data_bin_dir为转化后的的bin数据集路径

```
python3.7 jpg2bin.py --datasets_dir=${data_dir} --output_dir=${data_bin_dir}
```

## 2.推理

使用msame推理工具，发起推理测试，可参考 masame_inference.sh 

```
input_path=/  输入数据集路径
output_path=/ 推理输出路径 
sh msame_inference.sh ${input_path} ${output_path}

```



## 3.推理结果后处理
推理结果为二进制.bin格式数据，需要将其转换为可视的.jpg格式图片。

```
input_dir=/输入.bin数据路径
output_dir=/输出路径
python3.7 bin2jpg.py --data_dir=${input_dir}   --dst_dir=${output_dir}
```
## 4.推理样例
horse to zebra :  
![pic](./imgs/n02381460_120.jpg) ![pic](./imgs/n02381460_120_out.jpg)  

zebra to horse:  

![pics](./imgs/n02391049_4730.jpg)![pic](./imgs/out85.jpg)  
![pics](./imgs/n02391049_9680.jpg)![pic](./imgs/out135.jpg)


## 5.离线推理性能

* GPU 离线推理

GPU离线推理单个图片耗时约3898ms
![pics](./imgs/inference_gpu_sc.png)

* Ascend310 离线推理  

NPU离线推理单个图片耗时约1185ms
![pics](./imgs/inference_sc.png)
