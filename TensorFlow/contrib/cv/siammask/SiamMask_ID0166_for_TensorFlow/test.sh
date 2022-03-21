dataset="VOT2016"
#dataset="VOT2018"
gpu=0
platform="NPU"

      


for((i=1;i<=2;i++));
do
        model="./logs/ckpt/siamMask_model.ckpt-"$i"000"
        echo $model
        echo $dataset

        

        CUDA_VISIBLE_DEVICES=$gpu python3  ./tools/test.py \
            --config ./config/config.json \
            --resume $model \
            --mask \
            --dataset $dataset \
            --platform $platform

        python3 tools/eval.py --dataset $dataset --tracker_prefix C --result_dir ./test/$dataset
done

