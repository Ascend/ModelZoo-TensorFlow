MODEL="./model/srnet.om"
INPUT_I_S="./input_img/i_s"
INPUT_I_T="./input_img/i_t"
OUTPUT_BIN="./output_bin"
OUTPUT_IMG="./output_img"

if [ ! -f $MODEL ];then
    source atc.sh
fi

if [ ! -d "$OUTPUT_BIN" ]; then
    mkdir $OUTPUT
fi

./msame --model $MODEL \
	--input "${INPUT_I_S},${INPUT_I_T}" \
	--output $OUTPUT_BIN \
	--outfmt BIN

wait

for filename in `ls $OUTPUT_BIN` ; do
	mv "${OUTPUT_BIN}/${filename}" "$(echo "${OUTPUT_BIN}/${filename}" | sed 's/_output_0//')"
done

"python3.6" bin2img.py -s $OUTPUT_BIN -d $OUTPUT_IMG
	      
		 
