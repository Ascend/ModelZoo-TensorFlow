#! /bin/bash
model="./omModel/depthNet_tf.om"

#start dev
dev_input="./data_bin/dev"
dev_output="./feat_bin/dev"
rm -rf $dev_output/*
/root/AscendProjects/tools/msame/out/msame --model $model --input $dev_input --output $dev_output

#start test
test_input="./data_bin/test"
test_output="./feat_bin/test"
rm -rf $test_output/*
/root/AscendProjects/tools/msame/out/msame --model $model --input $test_input --output $test_output

# ACER(2.7% in paper. The smaller value means better performance)
dev_output="./feat_bin/dev/20210702_165221"
test_output="./feat_bin/test/20210702_165250"
python3.7.5 feat_bin_acer.py $dev_output $test_output

