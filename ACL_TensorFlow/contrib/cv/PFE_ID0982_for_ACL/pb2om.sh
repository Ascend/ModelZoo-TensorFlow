export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB=./pb_model/pb_pfe01-19_16-52.pb


atc \
--model=${PB} \
--framework=3 \
--output=./om_pfe \
--soc_version=Ascend310 \
--input_shape="input:1,112,96,3" \
--log=info \
--out_nodes="mu:0;sigma_sq:0" \
--precision_mode=allow_fp32_to_fp16 \
--op_select_implmode=high_precision
