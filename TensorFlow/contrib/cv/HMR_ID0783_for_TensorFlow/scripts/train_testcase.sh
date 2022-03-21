# TODO: Replace with where you downloaded the resnet_v2_50.
PRETRAINED=/home/data/zhanghy/hmr_models/resnet_v2_50.ckpt
# TODO: Replace with where you downloaded the smpl model.
SMPL=/home/data/zhanghy/hmr_models/neutral_smpl_with_cocoplus_reg.pkl
# TODO: Replace with where you downloaded the smpl mesh faces.
SMPL=/home/data/zhanghy/hmr_models/smpl_faces.npy
# TODO: Replace with where you downloaded the preprocessed data.
DATA_DIR=/home/data/zhanghy/hmr_datasets

CMD="python3 -u -m src.main \
--d_lr 1e-4 \
--e_lr 1e-5 \
--log_img_step 1000 \
--pretrained_model_path=${PRETRAINED} \
--smpl_model_path=${SMPL} \
--smpl_face_path=${SMPL_FACE} \
--data_dir ${DATA_DIR} \
--e_loss_weight 60. \
--batch_size=64 \
--use_3d_label True \
--e_3d_weight 60. \
--datasets lsp,lsp_ext,mpii,coco,mpi_inf_3dhp \
--epoch 1 \
--log_dir logs"

echo $CMD
$CMD