# TODO: Replace with path to the pb model you want to evaluate.
PB=pb_model/inference.pb

# TODO: Replace with where you downloaded the preprocessed data.
DATA_DIR=/home/zhanghy/storage/hmr_datasets

CMD="python3 -u -m src.inference_from_pb \
--pb_path=${PB} \
--test_data_dir=${DATA_DIR}/mpi_inf_3dhp/test \
--batch_size=1"

echo $CMD
$CMD