# EfficientNets

This repository provides a script and recipe to train the efficientnetb4 model. 

## Requirement

tensorflow==1.15 

## Datasets

ImageNet2012

## Training Command Reference

```
cd test
bash env.sh
bash train_performance_1p.sh --train_steps=200 --data_path=/data/imagenet_TF --ckpt_path=ckpt --mode=train --train_batch_size=64 --iterations_per_loop=100
```

## Command-line options 

```
python3.7 ${currentDir}/main_npu.py \
    --data_dir=/data/slimImagenet \
    --model_dir=./ \
    --mode=train \
    --train_batch_size=64 \
    --train_steps=100 \
    --iterations_per_loop=10 \
    --model_name=efficientnet-b4 
```

## Performance

```
Final Performance ms/step : 407.94
Final Training Duration sec : 1005
```

##  Accuracy
