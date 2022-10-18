export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./log/pb_file/result.pb --framework=3 --output=./log/om_file/result --soc_version=Ascend910 \
        --input_shape="x_in:100,250" \
        --log=info \
        --out_nodes="loss_i:0"
