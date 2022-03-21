# TODO: Replace with path to the pb model you want to convert.
PB=./pb_model/monoculartotalcapture.pb


atc \
--model=${PB} \
--framework=3 \
--output=./om_model \
--soc_version=Ascend310 \
--input_shape="input:1,368,368,3" \
--log=info \
--out_nodes="output0:0;CPM/out_11:0;output2:0" \
--precision_mode=allow_fp32_to_fp16 \
--op_select_implmode=high_precision
