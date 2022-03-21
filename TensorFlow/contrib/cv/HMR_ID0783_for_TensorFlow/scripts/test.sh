# TODO: Replace with path to the model you want to evaluate.
LP=/home/data/zhanghy/hmr_models/model.ckpt-667589

# TODO: Replace with where you downloaded the smpl model.
SMPL=/home/data/zhanghy/hmr_models/neutral_smpl_with_cocoplus_reg.pkl

# TODO: Replace with where you downloaded the preprocessed data.
DATA_DIR=/home/data/zhanghy/hmr_datasets

CMD="python3 -u -m src.eval \
--load_path=${LP} \
--smpl_model_path=${SMPL} \
--eval_data_dir=${DATA_DIR}/mpi_inf_3dhp/test"

echo $CMD
$CMD