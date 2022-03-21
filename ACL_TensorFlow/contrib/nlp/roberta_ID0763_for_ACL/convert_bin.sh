BASE_DIR=/home/test_user02/lil/output/msame/squadv1 #原始输入文件夹
TF_DIR=/home/test_user02/lil/data/SQuAD/eval/squadv1 #tf文件路径

#多输入的.bin格式数据
input_id_path=$BASE_DIR/input_ids
input_mask_path=$BASE_DIR/input_masks
segment_id_path=$BASE_DIR/segment_ids

if [ ! -d ${input_id_path} ]; then
  mkdir ${input_id_path}
fi
if [ ! -d ${input_mask_path} ]; then
  mkdir ${input_mask_path}
fi
if [ ! -d ${segment_id_path} ]; then
  mkdir ${segment_id_path}
fi

python3.7 convert_bin.py --base_dir ${BASE_DIR} --tf_dir ${TF_DIR} --max_seq_length=512