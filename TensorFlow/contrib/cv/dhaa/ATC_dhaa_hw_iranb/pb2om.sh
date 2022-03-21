atc \
    --input_shape="image_batches:1,224,224,3" \
    --check_report=./pbModel/network_analysis.report \
    --input_format=NHWC \
    --output="/root/modelzoo/DHAA_tf/device/DHAA_tf" \
    --soc_version=Ascend310 \
    --framework=3 \
    --model="/root/code/dhaa/LAJ_DHAA/dhaa/dhaa_tf_hw80211537/pbModel/DHAA_tf.pb" \

