export output_dir=./raw_data
mkdir  ${output_dir}
rm -rf ${output_dir}/*

cd ${output_dir}
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz

cd ../
cd preprocess

python3 1_convert_pd.py
python3 2_remap_id.py

echo "preprocess data is done!"