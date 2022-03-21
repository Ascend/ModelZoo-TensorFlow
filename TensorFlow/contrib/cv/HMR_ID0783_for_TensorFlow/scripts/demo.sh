# TODO: Replace with path to the input image.
IMG=demo/coco1.png

# TODO: Replace with path to the model you want to run.
LP=/home/data/zhanghy/hmr_models/model.ckpt-667589

# TODO: Replace with where you downloaded the smpl model.
SMPL=/home/data/zhanghy/hmr_models/neutral_smpl_with_cocoplus_reg.pkl

# TODO: Replace with where you downloaded the smpl mesh faces.
SMPL_FACE=/home/data/zhanghy/hmr_models/smpl_faces.npy

# TODO: Replace with where you want to save the output.
SAVE=demo/output

CMD="python3 -u -m src.demo \
--img_path=${IMG} \
--load_path=${LP} \
--smpl_model_path=${SMPL} \
--smpl_face_path=${SMPL_FACE} \
--save_path=${SAVE}
"

echo $CMD
$CMD