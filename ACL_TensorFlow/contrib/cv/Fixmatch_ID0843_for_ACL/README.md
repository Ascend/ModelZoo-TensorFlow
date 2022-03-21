# FixMatch

Code for the paper: "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by 
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.

This is not an officially supported Google product.

![FixMatch diagram](media/FixMatch%20diagram.png)

## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install imagemagick
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

### Note

We provide a small and convenient dataset, this dataset only offer cifar10 dataset and specify that the `seed=5`, 40 labeled samples and 1 validation sample, you can use it easily by download [here](https://pan.baidu.com/s/1zuoA2-3E6H8ArEFC3XZM9w), keyword: tr3t, and put the folder in project root directory.

### ImageNet

Codebase for ImageNet experiments located in the [imagenet subdirectory](https://github.com/google-research/fixmatch/tree/master/imagenet).

## Preparation

### Environment configuration

Please config the environment setting after you convert model to OM, and use OM model to inference.  
The method of configuring environment variables can refer to here: [use ATC to convert model](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html) and [use msame tool to inference](https://gitee.com/ascend/tools/tree/master/msame)
 
### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Convert INPUT to BIN

```bash
python convertIMG_2_BIN.py --batchsize=1
```

This command will create floder named `input_bin_{batchsize}` for inference bin file, and `output_label_{batchsize}` for TRUE label of the input bin file. the `batchsize` in command can use [1,4,8,16,32], if you want to change the dataset. you can use `--dataset=XXX`

### ATC command 

You can use the bash file `ATC_PB_2_OM.sh` in test to finish model conversion. for example:

```bash
bash ATC_PB_2_OM.sh --model=/home/HwHiAiUser/fixmatch_npu.pb --output=/home/HwHiAiUser/fixmatch310_final --dynamic_batch_size="1,4,8,16,32"
```

If you want to see the help message of the bash file, you can use:

```bash
bash ATC_PB_2_OM.sh --help
```

You will see the help and the default setting of the args.

## Inference

You can use `OM_INFER.sh` in test to finish the inference and output the result.

```bash
bash OM_INFER.sh --masme_path=/home/HwHiAiUser/msame/tools/msame/out --model=/home/HwHiAiUser/fixmatch310_sh.om --input=/home/HwHiAiUser/mix_input --output=/home/HwHiAiUser/msame/out/ --dymBatch=1
```

args:  
`--masme_path` : Select the masme tool place where included the msame file after build.  
`--input` : Select the folder or bin file for input    
`--output`: Select the place of result folder  
`--dymBatch`: Select the dynamic_batch_size, the range is dynamic_batch_size in the ATC command.  

If you want to see the help message of the bash file, you can use:

```bash
bash OM_INFER.sh --help
```

You will see more help and the default setting of the args.

## Accuracy

### Calculate the inference accuracy

We offer a python file named `cal_inference_pref.py` to calculate the accuracy of inference.

```bash
python cal_inference_pref.py --PREDICT_LABEL_FLODER=XXX --OUTPUT_LABEL_FLODER=XXX 
```

you must select the floder of the `PREDICT_LABEL_FLODER` which is result of `OM_INFER.sh` and the `OUTPUT_LABEL_FLODER` which is created by `convertIMG_2_BIN.py`

**Note:** the result in `PREDICT_LABEL_FLODER` and `OUTPUT_LABEL_FLODER` must have the same `batch_size` 

### Accuracy

| Image nums | Accuracy Train | Accuracy Infer |
| :----:| :----: | :----: |
| 10000 | 0.8709  |  0.8708 |


## Appendix
+ Train_ACC shows in project: `Fixmatch_ID0843_for_TensorFlow`
+ PB file that has been generated can download by [here](https://pan.baidu.com/s/1kUvZ-2Fdv3Eb7KJqiTT4Dg), keyword: ra8d
+ OM file that has been generated can download by [here](https://pan.baidu.com/s/1msWWPbz8b2NO-QFioavxKg), keyword: fpp8
This OM model is generated by the following parameters：  

```bash
--dynamic_batch_size="1,4,8,16,32"
--soc_version=Ascend310
--input_shape="x:-1,32,32,3"
--out_nodes="Softmax_2:0"
```




