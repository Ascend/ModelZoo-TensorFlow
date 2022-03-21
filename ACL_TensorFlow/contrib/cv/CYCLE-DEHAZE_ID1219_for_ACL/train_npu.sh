export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)
export TF_CPP_MIN_LOG_LEVEL=0                   ## Tensorflow api print Log Config
code_dir=${1}
data_dir=${2}
result_dir=${3}
logs_dir=${4}
checkpoints_dir=${5}
models_dir=${6}
obs_url=${7}
profiling_dir=${8}

echo 'start to train!'
today="$(date '+%d_%m_%Y_%T')"

## 3.train according to the tfrecords
python3.7 ${code_dir}/train_npu.py \
        --dataset=${data_dir} \
        --result=${result_dir} \
        --logs=${logs_dir} \
        --checkpoints=${checkpoints_dir} \
        --models=${models_dir} \
        --obs_dir=${obs_url} \
        --batch_size 2 \
        --image_size1 256 \
        --image_size2 256 \
        --learning_rate 1e-4 \
        --X ${data_dir}/hazyImage.tfrecords \
        --Y ${data_dir}/clearImage.tfrecords \
        --log_file ${logs_dir}/train_npu_${today}.log \
        --profiling_dir ${profiling_dir}
echo 'training finished!'