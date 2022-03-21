#! /bin/bash
model="/root/code/dhaa/LAJ_DHAA/dhaa/dhaa_tf_hw80211537/om_model/om_model.om"
test_input="./data_bin/test"
test_output="./feat_bin/test"
rm -rf $test_output/*
/root/AscendProjects/tools/msame/out/msame --model $model --input $test_input --output $test_output