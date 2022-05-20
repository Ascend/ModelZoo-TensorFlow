
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_PATH=/root
OM_PATH=/root

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_PATH/gan.pb --framework=3 \
        --input_shape="split_1:64,128" \
	      --output=$OM_PATH/ganom --soc_version=Ascend310 \
        --out_nodes="generator/Sigmoid:0" \
	      --log=debug
