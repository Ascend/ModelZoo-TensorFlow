BASE_DIR=/home/test_user02/th/seq2seq_ID1474_for_ACL
BIN_DATA_DIR=$BASE_DIR/bin_data

encoder0_path=$BIN_DATA_DIR/encoder0
encoder1_path=$BIN_DATA_DIR/encoder1
encoder2_path=$BIN_DATA_DIR/encoder2
encoder3_path=$BIN_DATA_DIR/encoder3
encoder4_path=$BIN_DATA_DIR/encoder4
encoder5_path=$BIN_DATA_DIR/encoder5
encoder6_path=$BIN_DATA_DIR/encoder6
encoder7_path=$BIN_DATA_DIR/encoder7
encoder8_path=$BIN_DATA_DIR/encoder8
encoder9_path=$BIN_DATA_DIR/encoder9
encoder10_path=$BIN_DATA_DIR/encoder10
encoder11_path=$BIN_DATA_DIR/encoder11
encoder12_path=$BIN_DATA_DIR/encoder12
encoder13_path=$BIN_DATA_DIR/encoder13
encoder14_path=$BIN_DATA_DIR/encoder14
encoder15_path=$BIN_DATA_DIR/encoder15
encoder16_path=$BIN_DATA_DIR/encoder16
encoder17_path=$BIN_DATA_DIR/encoder17
encoder18_path=$BIN_DATA_DIR/encoder18
encoder19_path=$BIN_DATA_DIR/encoder19
encoder20_path=$BIN_DATA_DIR/encoder20
encoder21_path=$BIN_DATA_DIR/encoder21
encoder22_path=$BIN_DATA_DIR/encoder22
encoder23_path=$BIN_DATA_DIR/encoder23
encoder24_path=$BIN_DATA_DIR/encoder24
encoder25_path=$BIN_DATA_DIR/encoder25
encoder26_path=$BIN_DATA_DIR/encoder26
encoder27_path=$BIN_DATA_DIR/encoder27
encoder28_path=$BIN_DATA_DIR/encoder28
encoder29_path=$BIN_DATA_DIR/encoder29
encoder30_path=$BIN_DATA_DIR/encoder30
encoder31_path=$BIN_DATA_DIR/encoder31
encoder32_path=$BIN_DATA_DIR/encoder32
encoder33_path=$BIN_DATA_DIR/encoder33
encoder34_path=$BIN_DATA_DIR/encoder34
encoder35_path=$BIN_DATA_DIR/encoder35
encoder36_path=$BIN_DATA_DIR/encoder36
encoder37_path=$BIN_DATA_DIR/encoder37
encoder38_path=$BIN_DATA_DIR/encoder38
encoder39_path=$BIN_DATA_DIR/encoder39
decoder0_path=$BIN_DATA_DIR/decoder0

ulimit -c 0
/home/test_user02/th/seq2seq_ID1474_for_ACL/main --model $BASE_DIR/om_model/seq2seq.om \
  --input ${encoder0_path},${encoder1_path},${encoder2_path},${encoder3_path},${encoder4_path},${encoder5_path},${encoder6_path},${encoder7_path},${encoder8_path},${encoder9_path},${encoder10_path},${encoder11_path},${encoder12_path},${encoder13_path},${encoder14_path},${encoder15_path},${encoder16_path},${encoder17_path},${encoder18_path},${encoder19_path},${encoder20_path},${encoder21_path},${encoder22_path},${encoder23_path},${encoder24_path},${encoder25_path},${encoder26_path},${encoder27_path},${encoder28_path},${encoder29_path},${encoder30_path},${encoder31_path},${encoder32_path},${encoder33_path},${encoder34_path},${encoder35_path},${encoder36_path},${encoder37_path},${encoder38_path},${encoder39_path},${decoder0_path} \
  --output $BASE_DIR/msame_out/ \
  --outfmt TXT \
  --device 1
