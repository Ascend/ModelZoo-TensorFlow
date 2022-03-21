
./obsutil cp obs://sunshk/bin_input_test.tar bin_input_test.tar
./obsutil cp obs://sunshk/bin_label_test.tar bin_label_test.tar
./obsutil cp obs://sunshk/assemble310.om assemble310.om

tar zxvf bin_input_test.tar
tar zxvf bin_label_test.tar
mkdir bin_predict_test


echo "prepare success!"

./msame --model assemble310.om --input bin_input_test --output bin_predict_test 
echo "msame success!"

cd code
python offline_infer.py --predict_path=../bin_predict_test --gt_path=../bin_label_test > inference.log 2>&1

if [ `grep -c "Accuracy" "inference.log"` -ne '0' ] ;then
	echo "Run testcase success!"
else
	echo "Run testcase failed!"
