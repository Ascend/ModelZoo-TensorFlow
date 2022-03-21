#直接运行Ascend310服务器自带的环境变量设置脚本
. /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/ascend-toolkit/5.0.2.alpha005/x86_64-linux/atc/bin/atc --model=/home/pb2om/Hazy2GT-bs2-dataset-80000.pb \
        --framework=3 \
        --output=/home/pb2om/om4/om4 \
        --soc_version=Ascend310 \
        --out_nodes="G_9/output/MirrorPad:0" \
        --log=info \
        --input_format=NHWC