CURRENT_DIR=$(cd$(dirname$0);pwd)

current_time=$(date +%Y%m%d%H%m%s)

mkdir ${CURRENT_DIR}/output

base_path=${CURRENT_DIR}/output/${current_time}

print_log="./inference_${current_time}.log"

mkdir ${base_path}
for i in {0..19};
do
    z_dir='z_'$i
    echo ${z_dir}
    /root/AscendProjects/tools/msame/out/msame --model /root/om_model/bicyclegan.om --input /root/dataset/input_temp,/root/ACL/data/z/${z_dir} --output ${base_path} --outfmt bin >> ${print_log} 2>&1 
done


# 推理速度计算
Generate_StepTime=`grep "Inference time: " ${print_log} | tail -n 7 | awk '{print $NF}' | awk -Fm '{print $1}'|awk '{sum+=$1} END {print sum/NR}'`
echo "Model Generate Images Perfomance sec/image:${Generate_StepTime}" >> ${print_log}

echo "Star evaluating"
python3 ./eval.py --output_path=${base_path} >> ${print_log} 2>&1
