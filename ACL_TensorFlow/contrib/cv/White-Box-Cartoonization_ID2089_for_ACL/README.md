# 模型功能
white-box-cartoonize 是一个将图像动漫化的模型。从图像中分别识别三种白盒表示：包含卡通图像平滑表面的表面表示，指赛璐珞风格工作流中稀疏色块和扁平化全局内容的结构表示，以及反映卡通图像中高频纹理、轮廓和细节的纹理表示。生成性对抗网络（GAN）框架用于学习提取的表示并对图像进行自动化。

- 参考论文：

    [Learning to Cartoonize Using White-box Cartoon Representations](https://github.com/SystemErrorWang/White-box-Cartoonization/tree/master/paper) 
    
    对于更详细的结果，可以参考[项目主页](https://github.com/SystemErrorWang/White-box-Cartoonization)

# pb模型冻结
在Ascend310推理服务器下或npu服务器上进行，由于需要使用到训练代码，因此需要将转换pb文件放在原代码code/train_code下。运行时将ckpt_path路径传入。
```bash
python3 ckpt2pb.py --ckpt_path=output/train_cartoon/saved_models
```
# om模型转换
在Ascend310推理服务器下进行om模型转化。
```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh #source环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=1

atc --model=/home/HwHiAiUser/AscendProjects/1/wbcnet.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/wbc/wbcnet --soc_version=Ascend310 --input_shape="input:1,256,256,3" --log=info --out_nodes="add_1:0"
```
请从此处下载[pb模型](https://canntf.obs.myhuaweicloud.com:443/vsl_zwt/vsl/pb_om/modelnet10.pb?AccessKeyId=NLVKVVAQHOUIA7ROJBEZ&Expires=1670766198&Signature=H4GOMDBr7ak8HGXRT4S03K/rJDc%3D)

请从此处下载[om模型](https://canntf.obs.myhuaweicloud.com:443/vsl_zwt/vsl/pb_om/modelnet10.om?AccessKeyId=NLVKVVAQHOUIA7ROJBEZ&Expires=1670766284&Signature=OuArsad0gLTjPmXi%2BPM4BbJUMYI%3D)


# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。


## 1.数据集转换bin
原始数据集包含了四类图片，在推理阶段需要将其转换为bin数据类型。在验证精度时我们仅使用scenery_photo

下载好原始数据集后，将其保存在dataset目录下，并执行`preprocess.py`文件将jpg数据转换为推理需要的[bin数据](obs://cann--id2089/dataset/scenery_photo/)


## 2.推理

使用msame推理工具，发起推理测试，推理命令如下：

```bash
./msame --model "gannet.om" --input "input_bin" --output "./output_final" --outfmt TXT
```

## 3.推理结果

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/AscendProjects/1/wbcnet.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/AscendProjects/VSL/out/modelnet10//20211215_215629
[INFO] start to process file:/home/HwHiAiUser/AscendProjects/1/input_bin/2013-11-08 16_45_24.jpg.bin
[INFO] model execute success
Inference time: 11.002ms
[INFO] get max dynamic batch size success
[INFO] output data success


Inference average time: 12.002000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

将推理生成的txt文件下载下来并存入`dataset`目录下，用于之后的精度测试。

## 4.后续处理
通过om文件生成的是bin文件，还需要通过后续处理转换成图片，运行process.py，生成的图片默认保存在当前文件的cartoonize文件下。若要修改则进入py文件内修改path变量即可。
```
python3 process.py
```
若要对比离线推理与在线训练结果，则将二者生成的图片读取，对比差距即可。也可以采用fid指标计算精度。