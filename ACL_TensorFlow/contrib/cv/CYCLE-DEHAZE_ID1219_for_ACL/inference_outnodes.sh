### ${1}为输入的图片数据集路径
### ${2}为pb模型推理结果的输出路径
### ${3}为推理使用的pb模型
if [ ! -d $2 ]; then
  mkdir $2
fi

for i in $1/*; do
	echo $i
	CUDA_VISIBLE_DEVICES="0" python3 inference_outnodes.py --model  $3 \
        	             			     --input_image  $i \
                	                             --output_path $2 \
                        	                     --image_size 256
done