# Inference for RFN

This folder provides a script to do inference with RFN using Ascend tool.

## Model Conversion

* Train the model or download the pre_trained model with the [BaiduNetDisk](https://pan.baidu.com/s/10im7mJNoQbjsUueQv3lLLw) (password: 3404).

* Generate frozen model with with the following commend:
```
 python frozen_pb.py --ckpt ./path/to/ckpt/folder/ --pb ./path/to/pb
```

* Convert pb model to om format with the [guide](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0047.html) or download the converted om model from the [BaiduNetDisk](https://pan.baidu.com/s/10im7mJNoQbjsUueQv3lLLw) (password: 3404).


## Inference Processing

### Data Pre-processing

Use the following commend to convert a image to bin format:
```
python to_bin.py --input ./path/to/image --output ./path/to/save/bin
```

### Model Inference

Install tool [msame](https://gitee.com/ascend/tools/tree/master/msame) and use the following commend to generate the redult of model inferance:
```
./msame --model ./path/to/om --input ./path/to/input/bin --output ./path/to/output/bin --outfmt BIN
```

### Data Post-processing

Use the following commend to generate super resolution image from the output BIN file:
```
python to_image.py --input ./path/to/output/bin --output ./path/to/output/image
```

## Result

* Input image:

<p align="center">
  <img src="./input.png" width=400px/> 
</p>

* Output image:

<p align="center">
  <img src="./output.png" width=400px/> 
</p>

