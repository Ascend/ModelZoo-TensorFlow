export TF_CPP_MIN_LOG_LEVEL=2                   ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0            ## Print log on terminal on(1), off(0)

code_dir=/home/test_user02/Arcface/prcode/ArcfaceCode
data_dir=/home/test_user02/Arcface/dataset
result_dir=/home/test_user02/Arcface/prcode/result


current_time=`date "+%Y-%m-%d-%H-%M-%S"`
python3.7 ${code_dir}/train_npu.py \
        --input_dir=${data_dir} \
        --result=${result_dir} \
        --code_dir=${code_dir} \
        --chip='npu' \
        --platform='apulis' \
        --npu_profiling=False  2>&1 | tee ${result_dir}/${current_time}_train_npu.log