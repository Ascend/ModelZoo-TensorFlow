OM_PATH=/root/infer
BIN_PATH=/root/infer/bin_data

/root/msame/out/./msame --model $OM_PATH/SEQUENCE_TAGGING.om --input $BIN_PATH/word_ids,$BIN_PATH/sequence_lengths,$BIN_PATH/char_ids --output $OM_PATH/