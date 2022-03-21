MODEL="/home/HwHiAiUser/modelzoo/ucf101_best/device/ucf101_best.om"
INPUT="/home/HwHiAiUser/econet/ucf101_bin"
OUTPUT="/home/HwHiAiUser/econet/ucf101_results_bin/"

cd $HOME/AscendProjects/tools/msame/out 

./msame --model $MODEL --input $INPUT --output $OUTPUT
