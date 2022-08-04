MODEL="./model/srnet.om"
PREDICT_DIR="./predict"
I_S_DIR="./predict/i_s"
I_T_DIR="./predict/i_t"
I_S_BIN_DIR="./predict/i_s_bin"
I_T_BIN_DIR="./predict/i_t_bin"
OUTPUT_BIN_DIR="./predict/output_bin"
OUTPUT_IMG_DIR="./predict/output_img"

function check_directory(){
    for dir_name in "$@"
    do
	if [ -f "$dir_name" ]; then
	    echo "warning: ${dir_name} is a file!"
    elif [ ! -d "$dir_name" ]; then
	    mkdir "$dir_name"
	fi
    done	
}

check_directory "$PREDICT_DIR" "$I_S_DIR" "$I_T_DIR" "$I_S_BIN_DIR" "$I_T_BIN_DIR" "$OUTPUT_BIN_DIR" "$OUTPUT_IMG_DIR"

if [ ! -f "$MODEL"]; then
    echo "om model doesn't exist, try to generate."
    source atc.sh
fi

python3 img2bin.py -i "$I_S_DIR" -h 64 -w 128 -f RGB -t float32 -m [127.5,127.5,127.5] -c [127.5,127.5,127.5] -o "$I_S_BIN_DIR"
python3 img2bin.py -i "$I_T_DIR" -h 64 -w 128 -f RGB -t float32 -m [127.5,127.5,127.5] -c [127.5,127.5,127.5] -o "$I_T_BIN_DIR"

wait

./msame --model "$MODEL" \
	--input "${I_S_BIN_DIR},${I_T_BIN_DIR}" \
	--output "$OUTPUT_BIN_DIR" \
	--outfmt BIN

wait

for filename in `ls $OUTPUT_BIN_DIR` ; do
	mv "${OUTPUT_BIN_DIR}/${filename}" "$(echo "${OUTPUT_BIN_DIR}/${filename}" | sed 's/_output_0//')"
done

python3 bin2img.py -s "$OUTPUT_BIN_DIR" -d "$OUTPUT_IMG_DIR"
