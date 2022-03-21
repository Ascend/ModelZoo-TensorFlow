export JOB_ID=10086
export ASCEND_DEVICE_ID=0
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

BERT_BASE_DIR=/home/test_user02/lil/model/Roberta-large
BASE_DIR=/home/test_user02/lil/output/msame/squadv1
TF_DIR=/home/test_user02/lil/data/SQuAD/eval/squadv1 #tf文件路径
CKPT_DIR=/home/test_user02/lil/data/ckpt/squadv1/model.ckpt-43800
SQUAD_DIR=/home/test_user02/lil/data/SQuAD/data

#rm -rf output

#训练模型转pb
python3 freeze_graph.py \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --output_dir=$BASE_DIR \
  --ckpt_dir=$CKPT_DIR \
  --max_seq_length=512

#pb转om
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$BASE_DIR/roberta.pb \
        --framework=3 \
        --output=$BASE_DIR/roberta \
        --soc_version=Ascend910 \
        --input_shape="input_ids:1,512;input_mask:1,512;segment_ids:1,512" \
        --log=info \
        --out_nodes="logits:0"


#数据转bin
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


input_id_path=$BASE_DIR/input_ids
input_mask_path=$BASE_DIR/input_masks
segment_id_path=$BASE_DIR/segment_ids
ulimit -c 0
/home/test_user02/lil/tools/msame/out/msame --model $BASE_DIR/roberta.om \
  --input ${input_id_path},${input_mask_path},${segment_id_path} \
  --output $BASE_DIR \
  --outfmt TXT

#rootdir需要自己检查推理文件路径后修改
python3 evaluate.py \
  --rootdir=$BASE_DIR/20210916_154109 \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --idx_file=$BASE_DIR/idx.txt \
  --output_dir=$BASE_DIR \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --max_seq_length=512 \
  --version_2_with_negative=False

python3 $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $BASE_DIR/predictions.json