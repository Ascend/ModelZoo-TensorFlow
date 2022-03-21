export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL/pb_model
OM_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL/om_model

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_DIR/seq2seq.pb \
        --framework=3 \
        --output=$OM_DIR/seq2seq \
        --soc_version=Ascend910 \
        --input_shape="encoder0:1;encoder1:1;encoder2:1;encoder3:1;encoder4:1;encoder5:1;encoder6:1;encoder7:1;encoder8:1;encoder9:1;encoder10:1;encoder11:1;encoder12:1;encoder13:1;encoder14:1;encoder15:1;encoder16:1;encoder17:1;encoder18:1;encoder19:1;encoder20:1;encoder21:1;encoder22:1;encoder23:1;encoder24:1;encoder25:1;encoder26:1;encoder27:1;encoder28:1;encoder29:1;encoder30:1;encoder31:1;encoder32:1;encoder33:1;encoder34:1;encoder35:1;encoder36:1;encoder37:1;encoder38:1;encoder39:1;decoder0:1" \
        --log=info
