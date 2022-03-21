# SESEMIF_for_TensorFlow_ACL

## 离线推理

### 离线推理命令参考
./msame --model="/root/sesemi/sesemi_inference_float32.om" --input="/root/sesemi/input_x/" --output="/root/sesemi/out/" --outfmt BIN
注意：输入数据保存至input_x文件夹中，文件路径需要更换

### .h5转.pb程序参考
利用软件对.h5文件转成.pb文件。软件下载地址如下：
链接：https://pan.baidu.com/s/1I23LztH5qhUSH-c9r_AD1g 
提取码：vbg4 

软件使用方法如下：
python keras_to_tensorflow.py --input_model="G:/HW/sesemi-master/model/sesemi_inference.h5" --output_model="G:/HW/sesemi-master/pb/sesemi_inference_new.pb"
注意：文件路径需要更换

### pb转om命令参考
atc --model=sesemi/pb/sesemi_inference_new.pb --framework=3 --output=sesemi/sesemi_inference_float32 --soc_version=Ascend310 --log=info --input_shape="super_data:10000,32,32,3"
注意：文件路径需要更换

### 输入数据转bin文件程序参考
运行DataToBin.py

### 推理结果

下载推理结果文件（下载地址在“推理过程中的文件”已经给出），并运行evaluate_tuili.py，得到推理精度
![输入图片说明](image.png)


总结：目前在训练样本数量为1000的条件下，GPU和NPU的精度均满足论文精度指标，推理精度略高于论文精度指标。

### 推理性能

[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /root/sesemi/sesemi_inference_float32.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/root/sesemi/out//20211222_155748
[INFO] start to process file:/root/sesemi/input_x//input_x_1.bin
[INFO] model execute success
Inference time: 1931.62ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_2.bin
[INFO] model execute success
Inference time: 1940.06ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_3.bin
[INFO] model execute success
Inference time: 1932.64ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_4.bin
[INFO] model execute success
Inference time: 1933.22ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_5.bin
[INFO] model execute success
Inference time: 1933.29ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_6.bin
[INFO] model execute success
Inference time: 1941.35ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_7.bin
[INFO] model execute success
Inference time: 1941.23ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/sesemi/input_x//input_x_8.bin
[INFO] model execute success
Inference time: 1933.29ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 1935.84 ms
Inference average time without first time: 1936.44 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl


### 推理过程中的文件

推理输入数据的获取地址如下：
链接：https://pan.baidu.com/s/1ecih8OOZIpD5KrnKjWtkDg 
提取码：q4n8 

离线训练网络（.h5, .pb, .om）的获取地址如下：
链接：https://pan.baidu.com/s/180MeyWPXixtFRf6_ACenQA 
提取码：p3la 

推理输出数据的获取地址如下：
链接：https://pan.baidu.com/s/12xGwsTH3Cuxg1POsCD2tkA 
提取码：4k7s 
