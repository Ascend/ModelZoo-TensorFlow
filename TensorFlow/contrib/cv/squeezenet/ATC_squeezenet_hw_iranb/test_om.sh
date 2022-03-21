#! /bin/bash
model="/root/code/Squeezent/squeezenet.om"
test_input="/root/dataset/imagenet/input"
test_output="/root/code/Squeezent/output_bin"
rm -rf $test_output/*
/root/AscendProjects/tools/msame/out/msame --model $model --input $test_input --output $test_output --debug true
