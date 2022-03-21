## 1、原始模型

参考训练工程，训练网络生成ckpt模型，并转成pb模型
```
https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/stargan/StarGAN_ID0725_for_TensorFlow
```
可通过链接 [ckpt](https://pan.baidu.com/s/1WaDxPPTgVmYLLF4Vc_deOg)
获取模型ckpt文件（提取码：fwq6）

可通过链接 [pb](https://pan.baidu.com/s/1DUMVIvxTu7G8Y9pAsPhqhA)
获取模型pb文件（提取码：vf58）


## 2、pb转om模型

atc转换命令参考：
```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc \
        --model=./stargan_model.pb \
        --framework=3 \
        --soc_version=Ascend310 \
        --input_format=NHWC \
        --input_shape="x_real:1,128,128,3;c_trg:1,5" \
        --log=info \
        --output_type=FP32 \
        --out_nodes="G/Tanh:0" \
        --output=./stargan_model
```
说明：上述命令中，指定的推理硬件为Ascend310，om模型推理时要在对应的硬件上运行；input_shape的设置，x_real,c_trg为输入节点名，batch需要设置为1；指定out_nodes的输出时，需要在节点名G/Tanh后加上:0，不然会报错；文件路径请根据情况配置。

可通过链接 [om](https://pan.baidu.com/s/1GGKIMUtzn9XVCR6TvzBFNQ)
获取模型om文件（提取码：1x2n）


## 3、数据转换bin文件

数据转换成bin文件，可使用numpy.array的tofile函数，模型输入为两条数据，将输入分别保存在不同的文件夹中，每个文件夹中包含多条测试样例，相同测试样例的不同输入文件夹下的文件名相同，如下：
```
    for idx in .....:
        x_real.tofile("./x_real/{0:05d}.bin".format(idx))
        c_trg.tofile("./c_trg/{0:05d}.bin".format(idx))
```


## 4、编译msame推理工具

参考 https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 5、使用msame进行离线打通

使用msame推理工具，参考如下命令，发起推理性能测试： 
```
./msame --model "./stargan_model.om" --output "./test_data/outputs_om" --outfmt BIN --loop 100
```

推理测试成功，会输出如下log信息：
```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./model/stargan_model.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
./test_data/outputs_om/20210831_144717
[INFO] model execute success
Inference time: 7.507ms
[INFO] model execute success
Inference time: 7.401ms
... ...
[INFO] model execute success
Inference time: 7.472ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 7.417150 ms
Inference average time without first time: 7.416242 ms
[INFO] destroy model input success.
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```
- 1Batch，shape:1x128x128x3，不带AIPP，平均推理性能 7.42ms
- 4Batch，shape:4x128x128x3，不带AIPP，平均推理性能 25.43ms
- 8Batch，shape:8x128x128x3，不带AIPP，平均推理性能 53.00ms
- 16Batch，shape:16x128x128x3，不带AIPP，平均推理性能 109.42ms
- 32Batch，shape:32x128x128x3，不带AIPP，平均推理性能 216.90ms


## 6、CelebA数据集上精度测试：

### 6.1 数据集准备

1. 模型训练使用**CelebA**数据集，CelebA是香港中文大学的开放数据，包含10177个名人身份的202599张图片，并且都做好了特征标记，这对人脸相关的训练是非常好用的数据集。
    
   [CelebA数据集官网](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  


2. 数据集请用户在官网自行获取图像img_align_celeba.zip和标签list_attr_celeba.txt文件，或通过如下链接获取：
  
   - 图像数据 [CelebA images](https://pan.baidu.com/s/1Y7rWZidpQJdzroqcOR7qxQ)
  (提取码：0000)

   - 标签数据 [CelebA attribute labels](https://pan.baidu.com/s/1y-Jk9U3Ki_cqCJwWlU_SrA)
  (提取码：0000)


3. 数据集下载后，放入相应目录下，在数据预处理脚本中指定数据集路径，可正常使用。

### 6.2 数据预处理

下载好的数据集放在任意目录，执行预处理脚本生成bin文件：
```
python3.7 img_process.py --phase img_preprocess --image_root ./datasets/celeba/images --metadata_path ./datasets/celeba/list_attr_celeba.txt --save_dir ./test_data
```

### 6.3 执行推理
```
./msame --model "./stargan_model.om" --input "./test_data/x_real","./test_data/c_trg" --output "./test_data/outputs_om" --outfmt BIN
```

### 6.4 数据后处理

执行预处理脚本生成bin文件转换成图像文件：
```
python3.7 img_process.py --phase img_postprocess --input_images_bin ./test_data/x_real --input_results_om_bin ./test_data/outputs_om --result_dir ./results_om
```

### 6.5 精度测试
1. 在CelebA数据集上抽取了2000张图像作为测试数据，通过FID量化评价指标评测模型精度。FID 为从原始图像的计算机视觉特征的统计方面的相似度来衡量两组图像的相似度，分数越低代表两组图像越相似，或者说二者的统计量越相似。FID相关实现可参见官方github工程: [FID](https://github.com/bioinf-jku/TTUR)
。
   

2. 参见requirements.txt配置测试所需依赖库
    ```
    tensorflow==1.15.0
    opencv-python>=4.2.0
    numpy>=1.81.0
    tqdm
    imageio
    ```

3. 下载FID精度测试用模型 [Inception model](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)
   并执行命令
   ```
    mkdir tmp
    cd tmp
    tar zxvf inception-2015-12-05.tgz
   ```

4. 推理精度测试
   ```
    python3 fid.py ./results_om/src_images/ ./results_om/generates/
   ```
   2000张CelebA测试数据上
   - om模型推理的FID精度为: 15.342
   - pb模型推理的FID精度为: 15.341
    

5. NPU与GPU推理精度对比
   
   2000张CelebA测试数据上
   - NPU上om模型推理的FID精度为: 15.342
   - GPU上pb模型推理的FID精度为: 15.285
