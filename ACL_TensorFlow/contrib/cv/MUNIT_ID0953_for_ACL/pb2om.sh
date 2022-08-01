/usr/local/Ascend/ascend-toolkit/set_env.sh	# source 环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=1	# 设置输出日志打屏

mkdir debug_info
atc --model=munit.pb \
	--framework=3 \
	--output=munit \
	--soc_version=Ascend310 \
	--input_shape="test_imageA:1,256,256,3;test_imageB:1,256,256,3;test_style:1,1,1,8" \
	--log=info \
	--out_nodes="outputA:0;outputB:0" \
    --precision_mode="allow_fp32_to_fp16" \
	--debug_dir=debug_info \
| tee pb2om.log