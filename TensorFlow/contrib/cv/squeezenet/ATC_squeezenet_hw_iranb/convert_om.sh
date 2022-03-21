
atc --input_shape="input_1:1,224,224,3" \
    --check_report=/root/modelzoo/squeezenet/device/network_analysis.report \
    --input_format=NHWC \
    --output="/root/code/Squeezent" \
    --soc_version=Ascend310 \
    --framework=3 \
    --model="/root/code/Squeezent/squeezenet.pb" 
