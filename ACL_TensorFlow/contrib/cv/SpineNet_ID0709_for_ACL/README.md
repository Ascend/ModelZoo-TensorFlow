
## 推理过程<a name="section1589455252218"></a>
环境
- Tensorflow 1.15
- python 3.7

1.  ckpt文件

- ckpt文件下载地址:
  
  https://model-arts-1.obs.cn-north-4.myhuaweicloud.com:443/spinenet_npu/spinenet_model/checkpoint.zip?AccessKeyId=1UP3DRH2MU6VAE7IWIJR&Expires=1685603391&Signature=aSj4UjJQOXvZpqLKrndIK3vckUY%3D
  
    通过freeze_graph.py转换成pb文件spinenet_tf_310.pb, 例如：
	python freeze_graph.py --model=retinanet --config_file="official/detection/configs/spinenet/spinenet49S_retinanet.yaml" \
  	  --params_override="{ train: { total_steps : 231500, train_batch_size : 4}, eval: { eval_batch_size : 4 } }" \
	  --checkpointpath=/home/HwHiAiUser/spinetnet/model.ckpt-231500 --output=/home/HwHiAiUser/spinetnet
  
- pb文件下载地址:
  
  https://model-arts-1.obs.cn-north-4.myhuaweicloud.com:443/spinenet_npu/spinenet_model/spinenet_tf_310.pb?AccessKeyId=1UP3DRH2MU6VAE7IWIJR&Expires=1685627041&Signature=TvJiX0WI0ddSuD1YjQmzB5KoGr0%3D
  
2.  om模型

- om模型下载地址:
  
  https://model-arts-1.obs.cn-north-4.myhuaweicloud.com:443/spinenet_npu/spinenet_model/spinenet_tf_310.om?AccessKeyId=1UP3DRH2MU6VAE7IWIJR&Expires=1685627092&Signature=H4/IZz4vkUNDef7Jchq/EjQA/Eg%3D
   
  
  使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
  atc --model=/home/HwHiAiUser/spinetnet/spinenet_tf_310.pb --framework=3 --output=/home/HwHiAiUser/spinetnet/spinenet_tf_310 --soc_version=Ascend310 \
        --input_shape="Placeholder:1,640,640,3" --log=info \
        --out_nodes="NumDetections:0;DetectionBoxes:0;DetectionClasses:0;DetectionScores:0;ImageInfo:0"
```

3.  使用msame工具推理
    
    参考 https://gitee.com/ascend/tools/tree/master/msame，获取msame推理工具及使用方法。

    获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。
    
    msame推理可以参考如下指令:
```
./msame --model "/home/HwHiAiUser/msame/spinenet_tf_310.om" --input "/home/HwHiAiUser/msame/data/" --output "/home/HwHiAiUser/msame/out/" --outfmt TXT
```
- 将测试集数据转为bin文件:
```
  python image2bin.py --image_file_pattern=/tmp/imagedata/*.jpeg --output_dir=/tmp/bindata/
```

- 测试数据bin文件下载地址:
  
  https://model-arts-1.obs.cn-north-4.myhuaweicloud.com:443/spinenet_npu/spinenet_model/data.zip?AccessKeyId=1UP3DRH2MU6VAE7IWIJR&Expires=1685628672&Signature=E0XSD7h5uYnsFHu7iXUWvawVHM8%3D
  
4.  性能测试

  测试结果txt文件下载地址：
  https://model-arts-1.obs.cn-north-4.myhuaweicloud.com:443/spinenet_npu/spinenet_model/spinenet.zip?AccessKeyId=1UP3DRH2MU6VAE7IWIJR&Expires=1685629396&Signature=i4Y%2BefOLlABvtYpqRbt2X6E7ApE%3D
  
  使用test_spinenet_om_data.py对推理完成后获得的txt文件进行测试，例如
  python test_spinenet_om_data.py
  --output="/home/HwHiAiUser/msame/out/20220606_222003"
  --label_map_file="official/detection/datasets/coco_label_map.csv"
  --output_html="/home/HwHiAiUser/msame/out/test_spinenet_om.html"
  --image_file_pattern="/home/HwHiAiUser/msame/data/0*.jpeg"
 
	

<h2 id="精度测试">精度测试</h2>

训练集：coco2017

测试集：coco2017 val

  |精度指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |mAP|xxx|0.198|0.16|

  |性能指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |FPS|XXX|0.24 sec/batch|1.2 sec/batch|