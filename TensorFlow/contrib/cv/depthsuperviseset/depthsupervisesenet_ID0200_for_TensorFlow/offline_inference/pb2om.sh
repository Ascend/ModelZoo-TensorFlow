#atc --model=./pbModel/saved_model.pb --input_shape="degradated_image:1,256,256,3" --framework=3 --output=./omModel  --soc_version=Ascend310

atc --input_shape="color:1,256,256,3" \
--check_report=./omModel/network_analysis.report \
--input_format=NHWC \
--output=./omModel/depthNet_tf \
--framework=3 \
--model=./pbModel/depthNet_tf.pb \
--output_type=FP32 \
--soc_version=Ascend310 \

