
input_path=${1}
#input_path=/home/HwHiAiUser/arcface/data_bin/lfw/data_0.bin
output_path=${2}
#output_path=/home/HwHiAiUser/arcface/test1
ulimit -c 0
cd /home/HwHiAiUser/AscendProjects/tools/msame/out
./msame --model /home/HwHiAiUser/omfile/arcface_tf_310.om --input ${input_path} --output ${output_path} 