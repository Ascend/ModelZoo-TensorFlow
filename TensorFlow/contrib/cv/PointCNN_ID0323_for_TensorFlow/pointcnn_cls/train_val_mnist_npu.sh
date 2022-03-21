#!/usr/bin/env bash

npu=
setting=
models_folder="../../models/cls/"
train_files="../../data/mnist/train_files.txt"
val_files="../../data/mnist/test_files.txt"

usage() { echo "train/val pointcnn_cls with -g id -x setting options"; }

npu_flag=0
setting_flag=0
while getopts g:x:h opt; do
  case $opt in
  g)
    npu_flag=1;
    npu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $npu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ ! -d "$models_folder" ]
then
  mkdir -p "$models_folder"
fi


echo "Train/Val with setting $setting on $gpu!"
python3.7 ../train_val_cls.py -t $train_files -v $val_files -s $models_folder -m pointcnn_cls -p NPU -x $setting > $models_folder/pointcnn_cls_$setting.txt
