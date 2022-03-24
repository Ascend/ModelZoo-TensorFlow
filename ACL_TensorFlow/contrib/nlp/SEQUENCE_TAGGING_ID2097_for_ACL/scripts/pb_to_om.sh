export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_PATH=/root/infer
OM_PATH=/root/infer

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_PATH/SEQUENCE_TAGGING.pb --framework=3 \
        --output=$OM_PATH/SEQUENCE_TAGGING --soc_version=Ascend310 \
        --input_shape="word_ids:1,128;sequence_lengths:1;char_ids:1,128,64" \
        --out_nodes="dense/BiasAdd:0;transitions:0"