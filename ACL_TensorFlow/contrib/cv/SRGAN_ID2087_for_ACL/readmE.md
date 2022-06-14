## 模型功能

 将低分辨率图片转化到高分辨率

整个过程用的推理图片为t480.jpg
obs地址:obs://ksgannn/tuili/t480.jpg
## h5模型
```
gen_model500.h5
```
obs地址：obs://ksgannn/tuili/gen_model500.h5

步骤一:将gen_model.h5转化成npuzh.pb
需要通过代码h5_pb.py将h5转成pb（将gen_model500.h5转换成npuzh.pb）
h5_pb.py代码已提交

## pb模型

```

npuzh.pb
```

npuzh.pb（已修改好输入的） 
obs地址：obs://ksgannn/tuili/npuzh.pb
步骤二：由于转换的pb模型的输入是-1,48,48,3的，在转换到om时会有问题，所以需要手动修改输入维度到1，48，48，3（batch数值可以修改，推理过程为了检验模型是否正确都用batch为1
    通过代码pb_pbtxt.py将pb修改成pbtxt，然后把pbtxt的输入维度从-1,48,48,3修改至1,48,48,3， 再通过代码pb_pbtxt.py将pbtxt改回pb(注意路径) （pb_pbtxt.py内有两个函数 需要手动调执行pb转pbtxt还是pbtxt转pb）
pb_pbtxt.py代码已提交

## om模型

步骤三：转npuzh.pb到srgan.om
使用ATC模型转换工具进行模型转换时可以参考如下指令:


```
atc --model=./npuzh.pb 
    --framework=3 
    --output=./srgan 
    --soc_version=Ascend310 
    --input_shape="Input:1,48,48,3" 
    --log=info 
    --out_nodes="Identity:0"

```
成功转化成srgan.om
srgan.om的obs地址：obs://ksgannn/tuili/srgan.om

## 使用msame工具推理


参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

这里是cd ~/tools/msame/sragn  (需要在~/tools/msame下mkdir srgan && cd srgan)
把图片t480.jpg和srgan.om都放在srgan目录下

## 数据集转换bin

这里只进行一张图片转化测试性能
```
zhuanbin.py
```
步骤四：转换数据
zhuanbin.py已提交代码

使用方法:zhuanbin中也有两个函数 执行zhuanbin.py里的save2bin_1()可以将图片转化为预处理后的bin（见48.bin obs地址 obs://ksgannn/tuili/48.bin）
在推理完生成srgan_output_0.bin后 执行zhuanbin.py里的load_1()将生成的bin转回jpg形式（见opt.jpg  obs地址 obs://ksgannn/tuili/opt.jpg）

## 推理测试


使用msame推理工具，参考如下命令，发起推理测试：
 

```
./msame/srgan  执行../out/msame --model "./srgan.om" --input "./48.bin" --output "./output" --outfmt BIN --loop 1

```




## 推理精度
GPU和NPU推理出来的图片效果相似（srgan没有具体精度，主要是视觉看上去分辨率的提升）