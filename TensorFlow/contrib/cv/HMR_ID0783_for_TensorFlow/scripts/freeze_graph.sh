# TODO: Replace with path to the model you want to freeze.
LP=/home/data/zhanghy/hmr_models/model.ckpt-667589

# TODO: Replace with where you downloaded the smpl model.
SMPL=/home/data/zhanghy/hmr_models/neutral_smpl_with_cocoplus_reg.pkl

CMD="python3 -u -m src.freeze_graph \
--load_path=${LP} \
--batch_size=1 \
--smpl_model_path=${SMPL}"

echo $CMD
$CMD