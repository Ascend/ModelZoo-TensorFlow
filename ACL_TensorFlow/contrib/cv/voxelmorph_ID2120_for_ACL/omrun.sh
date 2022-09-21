export DDK_PATH=/home/TestUser02/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/home/TestUser02/Ascend/ascend-toolkit/latest/acllib/lib64/stub

input_src=../Dataset-ABIDE/test_bin/002.bin
input_tgt=../Dataset-ABIDE/tgt.bin
ulimit -c 0
/home/TestUser02/tools/msame/out/msame --model ./models/vm.om \
--input ${input_src},${input_tgt} --output ./output \
--outputSize "1000000000, 1000000000"  --outfmt BIN --loop 1 --debug true