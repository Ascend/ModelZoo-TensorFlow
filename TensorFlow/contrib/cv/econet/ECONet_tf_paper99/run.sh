run_mode=$1
dataset_name=$2

echo "Dataset: ${dataset_name}"

train_path="./splits_txt/${dataset_name}/${dataset_name}_train_split_1_rawframes.txt"
val_path="./splits_txt/${dataset_name}/${dataset_name}_val_split_1_rawframes.txt"
dataset_path="./data/${dataset_name}_extracted"

if [ $run_mode = 'train' ]
then
    echo "Training mode!"
    resume_path="./experiments/ECOfull/ECOfull_kinetics.ckpt"
    python3.7 train.py ${dataset_name} ${dataset_path} RGB ${train_path} ${val_path} --resume_path ${resume_path} 
elif [ $run_mode = 'test' ]
then
    echo "Test mode!"
    resume_path="./experiments/hmdb51_best/ckpt/best.ckpt"
    python3.7 test.py ${dataset_name} ${dataset_path} RGB ${train_path} ${val_path} --resume_path ${resume_path}
else
    echo "$run_mode mode is not supported."
fi
