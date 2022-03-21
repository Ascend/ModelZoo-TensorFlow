# Yolov5

YoloV5 implemented by TensorFlow2 , with support for training, evaluation and inference. <br>

> **NOT perfect** project currently, but I will continue to improve this, so you might want to watch/star this repo to revisit. Any contribution is highly welcomed<br>

![demo](./data/sample/demo1.png)

<!--
| Model | Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |  cfg | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| YOLOV5s | 672 | 47.7% |52.6% | 61.4% | 
| YOLOV5m | 672 | 47.7% |52.6% | 61.4% | 
| YOLOV5l | 672 | 47.7% |52.6% | 61.4% | 
| YOLOV5x | 672 | 47.7% |52.6% | 61.4% | 
|  |  |  |  |  |  |  |
-->

## Key Features
- minimal Yolov5 by pure tensorflow2
- yaml file to configure the model
- custom data training
- mosaic data augmentation
- label encoding by iou or wh ratio of anchor
- positive sample augment
- multi-gpu training
- detailed code comments
- full of drawbacks with huge space to improve

## Usage
### Clone and install requirements
```
$ git clone git@github.com:LongxingTan/Yolov5.git
$ cd Yolov5/
$ pip install -r requirements.txt
```
<!-- ### Download pretrained weights
```
$ cd weights/
$ bash download_weights.sh
``` -->
### Download VOC
```
$ bash data/scripts/get_voc.sh
$ cd yolo
$ python dataset/prepare_data.py
```

<!-- ### Download COCO
```
$ cd data/
$ bash get_coco_dataset.sh
``` -->
### Train
```
$ python train.py
```


### Inference
```
$ python detect.py
$ python test.py
```

### Train on custom data
If you want to train on custom dataset, PLEASE note the input data should like this:
```
image_dir/001.jpg x_min, y_min, x_max, y_max, class_id x_min2, y_min2, x_max2, y_max2, class_id2
```
And maybe new anchor need to be created, don't forget to change the nc(number classes) in yolo-yaml.
```
$ python dataset/create_anchor.py
```
## References and Further Reading
