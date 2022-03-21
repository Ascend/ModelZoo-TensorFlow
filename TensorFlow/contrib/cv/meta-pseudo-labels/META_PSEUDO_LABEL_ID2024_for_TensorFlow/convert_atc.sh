export PATH=/root/anaconda3/envs/delf/bin:$PATH
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/5.0.3.alpha003/atc/lib64/stub
atc --model=/root/projects/meta-pseudo-labels-tf1-gpu/pb_model/meta_pseudo_labels.pb --framework=3 --output=/root/projects/meta-pseudo-labels-tf1-gpu/pb_model/tf_meta_pseudo_labels --soc_version=Ascend910A --input_shape="input:1,32,32,3"
