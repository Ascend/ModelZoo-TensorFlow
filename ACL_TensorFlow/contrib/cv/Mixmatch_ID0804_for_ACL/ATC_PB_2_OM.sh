# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64
# export PATH=/usr/local/python3.7.5/bin:$PATH
# export PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
# export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
# export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

soc_version=Ascend910
input_shape="x:1,32,32,3"
out_nodes="Softmax_2:0"
dynamic_batch_size="1,4,8,16,32"
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:bash ./ATC_PB_2_OM.sh <args>"
    echo " "
    echo "parameter explain:
    --model                  set model place, e.g. /home/TestUser03/code/mixmatch/mix_model/mixmatch.pb
    --output                 set the name and place of OM model, e.g. /home/TestUser03/code/mixmatch/mix_model/mixmatch_om_final
    --soc_version            set the soc_version, default: --soc_version=Ascend910
    --input_shape            set the input node and shape, default: --input_shape="x:1,32,32,3"
    --out_nodes              set the out_nodes, default: --out_nodes="Softmax_2:0"
    --dynamic_batch_size     set the dynamic_batch_size, default: --dynamic_batch_size="1,4,8,16,32"
    -h/--help                show help message
    "
    exit 1
fi

for para in $*
do
    if [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    elif [[ $para == --output* ]];then
        output=`echo ${para#*=}`
    elif [[ $para == --soc_version* ]];then
        soc_version=`echo ${para#*=}`
    elif [[ $para == --input_shape* ]];then
        input_shape=`echo ${para#*=}`
    elif [[ $para == --out_nodes* ]];then
        out_nodes=`echo ${para#*=}`
    elif [[ $para == --dynamic_batch_size* ]];then
        dynamic_batch_size=`echo ${para#*=}`
    fi
done

if [[ $model  == "" ]];then
   echo "[Error] para \"model \" must be config"
   exit 1
fi

if [[ $output  == "" ]];then
   echo "[Error] para \"output \" must be config"
   exit 1
fi

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc \
                    --model=${model} \
                    --output=${output} \
                    --soc_version=${soc_version} \
                    --input_shape=${input_shape} \
                    --out_nodes=${out_nodes} \
                    --framework=3 \
                    --log=error \
#                    --dynamic_batch_size=${dynamic_batch_size} \
#                    --output_type=FP16