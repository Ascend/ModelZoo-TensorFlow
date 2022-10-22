# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64
# export PATH=/usr/local/python3.7.5/bin:$PATH
# export PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
# export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
# export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

outfmt=TXT
dymBatch=1
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:bash ./OM_INFER.sh <args>"
    echo " "
    echo "parameter explain:
    --msame_path             set the path of msame after build, e.g. --masme_path=/home/TestUser03/tools/msame/out
    --model                  set model place, e.g. --model="/home/TestUser03/code/mixmatch/mix_model/mixmatch_om_final.om"
    --input                  set the floder or file of BIN, e.g. --input="/home/TestUser03/code/mixmatch/mix_model/input_bin_01"
    --output                 set the place of inference result floder, e.g. --output="/home/TestUser03/code/mixmatch/mix_model/"
    --outfmt                 config the output file type, default: --outfmt=TXT
    --dymBatch               config the batchsize of the input, default: --dymBatch=1
    -h/--help                show help message
    "
    exit 1
fi

for para in $*
do
    if [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    elif [[ $para == --input* ]];then
        input=`echo ${para#*=}`
    elif [[ $para == --output* ]];then
        output=`echo ${para#*=}`
    elif [[ $para == --outfmt* ]];then
        outfmt=`echo ${para#*=}`
    elif [[ $para == --masme_path* ]];then
        masme_path=`echo ${para#*=}`
    elif [[ $para == --dymBatch* ]];then
        dymBatch=`echo ${para#*=}`
    fi
done

if [[ $masme_path  == "" ]];then
   echo "[Error] para \"masme_path \" must be config"
   exit 1
fi

if [[ $model  == "" ]];then
   echo "[Error] para \"model \" must be config"
   exit 1
fi

if [[ $input  == "" ]];then
   echo "[Error] para \"input \" must be config"
   exit 1
fi

if [[ $output  == "" ]];then
   echo "[Error] para \"output \" must be config"
   exit 1
fi

cd ${msame_path}
./msame --model=${model} \
        --input=${input} \
        --output=${output} \
        --outfmt=${outfmt} \
  #      --dymBatch=${dymBatch}