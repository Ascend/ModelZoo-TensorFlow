#!/bin/bash

test_datasets_dir=${1}
result_dir=${2}
model=${3}

if [ ! -d $result_dir ];then
  mkdir -p $result_dir
fi 

for file in `ls -a $test_datasets_dir`
do
  if [ "${file##*.}"x = "jpg"x ];then
      echo "start infer $file..."
      python3 ../inference.py \
      --model=$model \
      --input=$test_datasets_dir/$file \
      --output=$result_dir/"${file%.*}_out.jpg" \
      --image_size=256
  fi
done

