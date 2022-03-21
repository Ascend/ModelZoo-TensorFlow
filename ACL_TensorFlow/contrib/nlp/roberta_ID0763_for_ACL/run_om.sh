export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_DIR=/home/test_user02/lil/output/msame/squadv2

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_DIR/roberta.pb \
        --framework=3 \
        --output=$PB_DIR/roberta \
        --soc_version=Ascend910 \
        --input_shape="input_ids:1,512;input_mask:1,512;segment_ids:1,512" \
        --log=info \
        --out_nodes="logits:0"