## 模型功能

图像去模糊


## pb模型
```

python3.7 ckpt2pb.py
```
pb模型获取链接：

obs://dmsp-submit/离线推理/

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/usr/model_test/dmsp_frozen_model.pb 
--framework=3 
--output=/usr/model_test/dmsp_frozen_model
--soc_version=Ascend310 
--out_nodes="strided_slice_1:0" 
--input_shape "input_image:1,180,180,3"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

## 数据集转换bin

```
python3.7 demo_DMSP.py
```

## 推理测试

使用msame推理工具，参考如下命令，发起推理测试： 

```
./msame --model "/usr/model_test/dmsp/dmsp_frozen_model.om" 
--input "/usr/model_test/dmsp/dmsp_input_image.bin" 
--output "/usr/model_test/output/dmsp_out" 
--outfmt TXT  
--loop 1
```
但是该论文的方法比较特殊，一次去模糊需要多次调用模型预测，然后再计算偏移量来去模糊，所以把推理测试放到了python脚本中
```
 os.system('
 /home/HwHiAiUser/AscendProjects/tools/msame/out/msame --model "/usr/model_test/dmsp/dmsp_model.om" 
 --input "/usr/model_test/dmsp/dmsp_input_image.bin" 
 --output "/usr/model_test/output/dmsp_out" 
 --outfmt TXT  
 --loop 1')
```

脚本也可以直接对验证集进行推理输出,计算精度



## 推理精度

|gpu|npu|推理|
|:----:|:----:|:----:|
|26.06|26.06|26.06|


## 推理性能
batch_size：1
```
============start non-blind deblurring on Berkeley segmentation dataset==============
Initialized with PSNR: 19.36972284204723
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /usr/model_test/dmsp/dmsp_model.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/usr/model_test/output/dmsp_out/20220220_215201
[INFO] start to process file:/usr/model_test/dmsp/dmsp_input_image.bin
[INFO] model execute success
Inference time: 155.431ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 155.431000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
WARNING:tensorflow:From /usr/model_test/dmsp/DMSP-tensorflow-master/DMSPDeblur.py:162: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

Initialized with PSNR: 19.29085570826864
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /usr/model_test/dmsp/dmsp_model.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/usr/model_test/output/dmsp_out/20220220_220701
[INFO] start to process file:/usr/model_test/dmsp/dmsp_input_image.bin
[INFO] model execute success
Inference time: 155.722ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 155.722000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
Initialized with PSNR: 24.678145385947843
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /usr/model_test/dmsp/dmsp_model.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/usr/model_test/output/dmsp_out/20220220_222201
[INFO] start to process file:/usr/model_test/dmsp/dmsp_input_image.bin
[INFO] model execute success
Inference time: 156.499ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 156.499000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```