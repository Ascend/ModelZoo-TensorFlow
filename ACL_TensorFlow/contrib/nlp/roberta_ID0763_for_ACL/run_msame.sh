BASE_DIR=/home/test_user02/lil/output/msame/squadv2

input_id_path=$BASE_DIR/input_ids
input_mask_path=$BASE_DIR/input_masks
segment_id_path=$BASE_DIR/segment_ids
ulimit -c 0
/home/test_user02/lil/tools/msame/out/msame --model $BASE_DIR/roberta.om \
  --input ${input_id_path},${input_mask_path},${segment_id_path} \
  --output $BASE_DIR \
  --outfmt TXT