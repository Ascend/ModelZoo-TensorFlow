## Polygen模型离线推理

### 概述

- PolyGen是三维网格的生成模型，可顺序输出网格顶点和面。PolyGen由两部分组成：一个是顶点模型，它无条件地对网格顶点进行建模，另一个是面模型，它对以输入顶点为条件的网格面进行建模。顶点模型使用一个masked Transformer解码器来表示顶点序列上的分布。对于面模型，PolyGen将Transformer与pointer network相结合，以表示可变长度顶点序列上的分布。
- 参考文献： [[2002.10880\] PolyGen: An Autoregressive Generative Model of 3D Meshes (arxiv.org)](https://arxiv.org/abs/2002.10880)
- 参考实现： [https://github.com/deepmind/deepmind-research/tree/master/polygen](https://github.com/deepmind/deepmind-research/tree/master/polygen)

### Requirements

- tensorflow: 1.15.0
- Ascend 310
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- Model: Polygen

### 1.代码路径解释

```shell
 |-- bin                                  -- bin文件  
 |-- ckpt                                 -- checkpoint文件
 |  |--face
 |  |--vertex
 |-- out                                  -- 推理输出
 |  |--pb
 |  |--gpu
 |  |--om
 |-- pb                                   -- pb文件
 |-- label2bin.py                         -- 输入数据生成   
 |-- pb_validate.py                       -- 固化模型验证
 |-- pb_frozen_face.py                    -- face model固化
 |-- pb_frozen_vertex.py                  -- vertex model验证
 |-- pic2bin.py                           -- 将输入转为.bin文件
 |-- getAcc.py                            -- 精度验证脚本（推理结果后处理）
 |-- reqirements.txt                        
 |-- README.md  
```

### 2.将输入转为bin文件

- Polygen模型包含vertex model和face model，

```
    vertex model输入：标签context（label:shape[None,]）
    vertex model输出：生成顶点（shape:[None, None, 3]）和顶点mask（shape:[None, None])
    face model输入: vertex model输出
    face model输出: 生成face（shape:[None, None, 3])
```

- 运行label2bin.py文件将输入的标签label转为.bin文件

```shell
  python label2bin.py 
```

### 3.模型固化（转pb）

- 模型ckpt下载路径[Google Cloud ](https://console.cloud.google.com/storage/browser/deepmind-research-polygen)
- 模型pb和om下载路径链接：https://pan.baidu.com/s/1_3Zvh_DjgscGhLaZoCbsiw?pwd=8qhe 提取码：8qhe
- 模型训练完毕后会生成vertex model和face model两个ckpt，分别使用pb_frozen_vertex.py和pb_frozen_face.py固化模型

```shell
  python pb_frozen_vertex.py
  python pb_frozen_face.py  
```

得到vertex_model.pb和face_model.pb,进行验证

```shell
  python pb_validate.py
```

### 4. `ATC`模型转换（pb模型转om模型）

1. 请按照[`ATC`工具使用环境搭建](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)搭建运行环境。
2. 参考以下命令完成模型转换。

   ```shell
      atc --model=vertex_model.pb --framework=3 --output=./vertex_model --soc_version=Ascend310 --input_shape="v_input:2" --out_nodes="v_output:0"
      atc --model=face_model.pb --framework=3 --output=./face_model --soc_version=Ascend310 --input_shape="f_input:2,400,3;f_mask:2,400" --out_nodes="f_output:0"
   ```

   得到 vertex_model.om和face_model.om文件

### 5.离线推理

1. 请先参考https://gitee.com/ascend/tools/tree/master/msame，编译出msame推理工具
2. 在编译好的msame工具目录下执行以下命令。

   ```shell
   cd $HOME/tools/msame/out/
   ./msame --model "/home/test_user02/Polygen/pb/vertex_model.om" --input "/home/test_user02/Polygen/bin/label.bin" --output "/home/test_user02/Polygen/bin/" --outfmt BIN --loop 1 --outputSize "12800"
   python3 binsplit ## 将vertex model输出转换为face model的输入
   ./msame --model "/home/test_user02/Polygen/pb/face_model.om" --input "/home/test_user02/Polygen/bin/f_vertex.bin, /home/test_user02/Polygen/bin/f_mask.bin" --output "/home/test_user02/Polygen/bin/" --outfmt BIN --loop 1 --outputSize "16000"
   ```

   各个参数的含义请参考 https://gitee.com/ascend/tools/tree/master/msame

### 6.离线推理精度性能评估

- 运行getAcc.py文件得到pb和om推理结果的精度对比

```shell
  python getAcc.py --om_vertex ./out/om/vertex_model_output_0.bin  --om_face ./out/om/face_model_output_0.bin --pb_vertex ./out/pb/vertex_model.bin --pb_face ./out/pb/face_model.bin --gpu_vertex ./out/gpu/vertex_model.bin --gpu_face ./out/gpu/face_model.bin
```

参数解释：

```shell
--om_vertex        //顶点模型om推理输出路径
--om_face          //面模型om推理输出路径
--pb_vertex        //顶点模型pb推理输出路径
--pb_face          //面模型pb推理输出路径
--gpu_vertex       //顶点模型gpu输出路径
--gpu_face         //面模型gpu输出路径
```

**离线推理精度**：

经GPU训练后生成的图像与pb，om模型推理结果相比，相似度高，误差极小。

