source ~/env.sh

DATA_PATH_OUT="../data/output_file"
if [ ! -d "${DATA_PATH_OUT}" ]; then
  mkdir ${DATA_PATH_OUT}
fi

python3 preprocessing_smith.py --input_file=${DATA_PATH}/input_file/small_demo_data.external_wdp.filtered_contro_wiki_cc_team.tfrecord --output_file=${DATA_PATH}/output_file/smith_train_sample_input.tfrecord --vocab_file=${DATA_PATH}/uncased_L-12_H-768_A-12/vocab.txt