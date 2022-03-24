python train.py \
        --data_input_path=/root/hht/temp/dataset/DIV2K/DIV2K_train_LR_bicubic\
        --data_truth_path=/root/hht/temp/dataset/DIV2K/DIV2K_train_HR \
        --train_path=/root/hht/temp/result \
        --chip='gpu' \
        --model='bsrn' \
        --dataloader='div2k_loader' \
        --batch_size=8 \
	--log_freq=10 \
	--save_freq=100000 \
        --max_steps=1000000 \
        --platform='linux' \
        --scales='4' # 2,3,4 for choosing
#        --bsrn_clip_norm=${clip_norm}
