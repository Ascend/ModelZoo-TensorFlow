pip3 install opencv-python
pip3 install opencv-contrib-python

obsutil cp obs://cann-nju-srnet/vgg19_weights_tf_dim_ordering_tf_kernels_notop.pb ./model/ -f -r
obsutil cp obs://cann-nju-srnet/data/test/ ./model/testdata/ -f -r

bash train_testcase1000.sh > train.log

key1="the_max_iter"
key2=""
key3=""

if [`grep "$key1" "train.log"` -ne '0'] ;then
    echo `grep -c "$key1" "train.log"`
    echo "Run testcase success!"
else
    echo "Run testcase failed!"
fi
