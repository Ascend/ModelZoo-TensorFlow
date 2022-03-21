MODEL="/home/HwHiAiUser/pfe/om_pfe.om" # ATC转换得到的om模型文件
INPUT="/home/HwHiAiUser/pfe/input_bin" # 预处理完成后的bin文件所在目录
OUTPUT="/home/HwHiAiUser/pfe/" # 推理结果存储目录

./msame --model $MODEL --input $INPUT --output $OUTPUT
