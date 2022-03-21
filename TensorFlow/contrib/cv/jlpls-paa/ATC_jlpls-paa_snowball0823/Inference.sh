#! /bin/bash
model="./om_model/jspaa_om_model.om"

#start test
test_input="./data_bin/test"
test_output="./feat_bin/test"
# rm -rf $test_output/*
# /root/AscendProjects/tools/msame/out/msame --model $model --input $test_input --output $test_output

# dev_output="./feat_bin/dev/20210702_165221"
# test_output="./feat_bin/test/20210702_165250"
# python3.7.5 feat_bin_acer.py $dev_output $test_output

feature_bin_path="/root/code/LAJ_PED/ATC_jlplsjaa/feat_bin/test/20210709_140408"
test_dataset_path="/root/code/LAJ_PED/PETA"
cd /root/code/LAJ_PED/jlpls-paa/jlpls-paa_tf_hw80211537
python3.7.5 val_PadAttr_om_bin.py $feature_bin_path $test_dataset_path
