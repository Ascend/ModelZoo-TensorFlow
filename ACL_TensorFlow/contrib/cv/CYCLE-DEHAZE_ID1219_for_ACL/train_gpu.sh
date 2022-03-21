today="$(date '+%d_%m_%Y_%T')"

## batch_size: batch size
## image_size1: the height of image
## image_size2: the width of image
## learning_rate: learning rate
## X: the tfrecords file of hazy images
## Y：the tfrecords file of clear images
## log_file: log file
##train according to the tfrecords
python3 train_gpu.py --batch_size 2 \
                 --image_size1 256 \
                 --image_size2 256 \
                 --learning_rate 1e-4 \
                 --X data/tfrecords/hazyImage.tfrecords \
                 --Y data/tfrecords/clearImage.tfrecords \
                 --log_file logs/train_${today}.log