# TODO: Replace with where you downloaded the resnet_v2_50.
PRETRAINED=/home/data/zhanghy/hmr_models/resnet_v2_50.ckpt
# TODO: Replace with where you downloaded the smpl model.
SMPL=/home/data/zhanghy/hmr_models/neutral_smpl_with_cocoplus_reg.pkl
# TODO: Replace with where you downloaded the smpl mesh faces.
SMPL_FACE=/home/data/zhanghy/hmr_models/smpl_faces.npy
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
--epoch 300 \
--log_dir /home/data/zhanghy/logs"

# To pick up training from a previous model, set LP(just the directory which contains the checkpoint).
# LP=logs/HMR_3DSUP_coco-lsp-lsp_ext-mpi_inf_3dhp-mpii_CMU-jointLim_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_3dsup-weight60_Sep24_1654
# CMD="python3 -u -m src.main \
# --d_lr 1e-4 \
# --e_lr 1e-5 \
# --log_img_step 1000 \
# --load_path=${LP} \
# --smpl_model_path=${SMPL} \
# --smpl_face_path=${SMPL_FACE} \
# --data_dir ${DATA_DIR} \
# --e_loss_weight 60. \
# --batch_size=64 \
# --use_3d_label True \
# --e_3d_weight 60. \
# --datasets lsp,lsp_ext,mpii,coco,mpi_inf_3dhp \
# --epoch 300"

echo $CMD
$CMD