python -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=660 --train_subset=train \
        --seg_net_type=ENTROPY \
        --size_height=1408 \
        --size_width=512 \
        --with_seg_net=True \
        --with_decision_net=False \
        --storage_dir=output \
        --dataset_dir=db \
        --datasets=KolektorSDD-dilate=5 \
        --name_prefix=full-size_cross-entropy