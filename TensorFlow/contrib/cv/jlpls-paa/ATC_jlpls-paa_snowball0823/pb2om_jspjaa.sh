# atc --input_shape="color:1,256,256,3" \
# --check_report=./omModel/network_analysis.report \
# --input_format=NHWC \
# --output=./omModel/depthNet_tf \
# --framework=3 \
# --model=./pbModel/depthNet_tf.pb \
# --output_type=FP32 \
# --precision_mode allow_fp32_to_fp16 \
# --soc_version=Ascend310 \
export SLOG_PRINT_TO_STDOUT=1

atc --input_shape="image_batches:1,224,224,3" \
    --check_report=./om_model/network_analysis.report \
    --input_format=NHWC \
    --output="./om_model/jspaa_om_model" \
    --soc_version=Ascend310 \
    --framework=3 \
    --output_type=FP32 \
    --precision_mode allow_fp32_to_fp16 \
    --model="/root/code/LAJ_PED/pbModel/JSPJAA_tf.pb" \
    --op_debug_level  2 \
    --log="debug" >> ./om_model/to_om_log.txt