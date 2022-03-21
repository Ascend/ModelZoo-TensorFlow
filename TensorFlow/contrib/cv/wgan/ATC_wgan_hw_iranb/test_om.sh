#! /bin/bash
model="./om_model/wgan.om"
test_input="./input_bin"
test_output="./feat_bin"
rm -rf $test_output/*
/root/AscendProjects/tools/msame/out/msame --model $model --input $test_input --output $test_output --debug true