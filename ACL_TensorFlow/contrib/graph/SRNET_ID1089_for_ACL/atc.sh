source /usr/local/Ascend/ascend-toolkit/set_env.sh

/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/x86_64-linux/atc/bin/atc \
    --framework=3 \
    --model=./model/srnet.pb \
    --output=./model/srnet \
    --input_format=NHWC \
    --input_shape="input_t:1,64,128,3;input_s:1,64,128,3" \
    --out_nodes="o_f:0" \
    --soc_version=Ascend310

    #--input_fp16_nodes="input_t;input_s" \

    
