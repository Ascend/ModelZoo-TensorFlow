# ResNext101
This implements training of ResNext101 on the ImageNet dataset.

## ResNext101 Detail

Details, see code\resnext50_train\models\resnet50\resnet.py

## ResNext101 Config

code\resnext50_train\configs\

## Training

# 1p prefomance training 1p
bash test/train_performance_1p.sh

# 8p prefomance training 8p
bash test/train_performance_8p.sh

# 1p full training 1p
bash test/train_performance_1p.sh

# 8p full training 8p
bash test/train_performance_8p.sh


## ResNext101 training result


| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| :------: |    446    |   1      |   1      | O2       |
| 79.487   |   3494    |   8      |  121     | O2       |
