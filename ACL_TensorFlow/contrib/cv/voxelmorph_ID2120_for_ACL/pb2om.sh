export PYTHONPATH=/home/TestUser02/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/home/TestUser02/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

/home/TestUser02/Ascend/ascend-toolkit/latest/atc/bin/atc --input_shape="input_src:1,160,192,224,1;input_tgt:1,160,192,224,1" \
--out_nodes="spatial_transformer/map/TensorArrayStack/TensorArrayGatherV3:0;flow/BiasAdd:0" \
--output="./models/vm" \
--soc_version=Ascend910 --framework=3 --model="./models/frozen_model.pb" 