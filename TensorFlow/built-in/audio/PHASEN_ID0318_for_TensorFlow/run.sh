tar zxvf libsndfile-1.0.28.tar.gz
cd libsndfile-1.0.28
chmod +x *
./configure
make -j20
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/libsndfile-1.0.28/src/.libs/

data_path=/data1/turingDataset/02-Speech/ID0318_CarPeting_TensorFlow_PHASEN/

python3 nn_se/_2_train.py --data_path=$data_path
