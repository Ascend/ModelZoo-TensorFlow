CODE_PATH="./smith"
DATA_PATH="../data"
nohup python3 -m smith.run_smith --dual_encoder_config_file=${CODE_PATH}/config/dual_encoder_config.smith_wsp.32.48.pbtxt --output_dir=${DATA_PATH}/result_file/train_wsp_20220904/ --train_mode=finetune --num_train_steps=10000 --num_warmup_steps=1000 --schedule=train >> train_0904.log 2>&1 &