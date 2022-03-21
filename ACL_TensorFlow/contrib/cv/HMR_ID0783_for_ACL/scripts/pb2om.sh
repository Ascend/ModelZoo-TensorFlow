# TODO: Replace with path to the pb model you want to convert.
PB=pb_model/inference.pb


atc \
--model=${PB} \
--framework=3 \
--output=./om_model \
--soc_version=Ascend310 \
--input_shape="the_inputs:1,224,224,3" \
--log=info \
--out_nodes="the_outputs:0" \
--precision_mode=allow_fp32_to_fp16 \
--op_select_implmode=high_precision