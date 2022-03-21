# Inference for Keypointnet

This folder provides a script to do inference with keypointnet using Ascend tool.

## Model Conversion

* Train the model or download the pre_trained model with the [link](https://pan.baidu.com/s/1IWHL7ZLeHdoRuIN1lIAx-A) (password: wr43).

* Generate frozen model with with the following commend:
```
 python frozen_pb.py --ckpt ./path/to/ckpt/folder/ --pb ./path/to/pb
```

* Convert pb model to om format with the [guide](https://support.huaweicloud.com/usermanual-mindstudioc73/atlasmindstudio_02_0047.html) or download the converted om model from the [link](https://pan.baidu.com/s/1w-5ct_74k_AZO1JUskZcNw)(password: oml1).


## Inference Processing

### Data Pre-processing

Use the following commend to convert a image to bin format:
```
python image_to_bin.py --input ./path/to/image --output ./path/to/save/bin
```

### Model Inference

Install tool [msame](https://gitee.com/ascend/tools/tree/master/msame) and use the following commend to generate the redult of model inferance:
```
./msame --model ./path/to/om --input ./path/to/input/bin --output ./path/to/output/bin --outfmt TXT --loop 1
```

### Data Post-processing

Use the following commend to draw the detected keypoint into the raw input image:
```
python draw_point.py --bin ./path/to/output/bin --input ./path/to/input/image  --output ./path/to/output/image
```

## Result

* Input image:

<p align="center">
  <img src="./input.png" /> 
</p>

* Output image:

<p align="center">
  <img src="./output.png" /> 
</p>

