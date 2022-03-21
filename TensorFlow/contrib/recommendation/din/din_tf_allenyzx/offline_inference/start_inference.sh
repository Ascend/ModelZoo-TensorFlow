
embed_dim=8
maxlen=40
file=../raw_data/remap.pkl
input_checkpoint=../checkpoint/model.ckpt-19001
output_graph=../checkpoint/frozen_model.pb

#mkdir raw_data/
#rm -rf raw_data/*
#
## create test dataset
#python3 test_preprocess.py \
#--file=${file} \
#--embed_dim=${embed_dim} \
#--maxlen=${maxlen}


export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=${install_path}/tfplugin/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export TE_PARALLEL_COMPILER=8
export REPEAT_TUNE=False

# create pb model
python3 create_pb.py \
--maxlen=${maxlen} \
--input_checkpoint=${input_checkpoint} \
--output_graph=${output_graph} \
--device=npu

# inference
python3 inf_acc.py --pb_path=${output_graph}