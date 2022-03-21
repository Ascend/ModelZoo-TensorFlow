# TODO: Replace with where you downloaded the preprocessed data.
DATA_DIR=/home/zhanghy/storage/hmr_datasets

# TODO: Replace with path to save converted converted data.
CONVERTED=om_test_data

CMD="python3 -u -m src.convert_test_data \
--test_data_dir=${DATA_DIR}/mpi_inf_3dhp/test \
--save_dir=${CONVERTED}"

echo $CMD
$CMD