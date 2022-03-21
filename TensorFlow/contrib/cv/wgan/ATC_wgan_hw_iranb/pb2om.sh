export SLOG_PRINT_TO_STDOUT=1
atc --input_shape="z:1,100" \
    --check_report=./om_model/network_analysis.report \
    --input_format=NHWC \
    --output="./om_model/wgan" \
    --soc_version=Ascend310 \
    --framework=3 \
    --output_type=FP32 \
    --precision_mode allow_fp32_to_fp16 \
    --model="./pbModel/wgan.pb" \
    # --op_debug_level  2 \
    # --log="debug" >> 