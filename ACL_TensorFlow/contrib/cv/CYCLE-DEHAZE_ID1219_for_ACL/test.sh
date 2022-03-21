## 1. Use the trained model to dehazy the test images
echo "Testing started, please wait.."
set -e
today="$(date '+%d_%m_%Y_%T')"
#等同于 today=$(date +%d_%m_%Y_%T)，双引号、单引号都可以没有，+号必须有，否则无法调用date命令

pathInput=${1%/}     #测试数据读取路径，png文件
pathOutput=${2%/}    #模型预测的输出路径，png文件
modelfile="$3"       #模型文件路径，pb文件

path_downscaled="$pathOutput/temp"
path_output="$pathOutput"

if [ ! -d $path_output ]; then
  mkdir $path_output
fi

if [ ! -d $path_downscaled ]; then
  mkdir $path_downscaled
fi


#Create log file
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>logs/log_$today.out 2>&1

#Downscaling
command="resize_im('$pathInput','$path_downscaled')"
matlab -nojvm -nodesktop -r $command

#Dehazing
sh convertHazy2GT.sh "$path_downscaled" "$modelfile"

#Upscaling
command="laplacian('$path_downscaled','$pathInput','$pathOutput')"
matlab -nojvm -nodesktop -r $command

if [ -d $path_downscaled ]; then
  rm -rf $path_downscaled
fi


echo "test finished"


## 2.calculate metrics between the predict images and groundtruth images
echo 'start to calculate metrics between the predict images and groundtruth images!'
python3 cal_metrics.py --model_predict_dir data/testData/model_predict \  #模型预测的结果目录
                 --groundtruth_dir data/testData/groundtruth   #实际上的清晰图像目录
echo 'calculate metrics finished!'
