# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""  transform .jpg picture to .bin format.

python3 jpg2bin.py --XtoY 'testA' --YtoX 'testB' \
            --datasets_dir '/home/ma-user/modelarts/inputs/data_url_0/avatar_data' \
            --output_dir '/home/ma-user/modelarts/outputs/train_url_0/bin'
"""
import os
import tensorflow as tf
from utils import load_test_data
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--XtoY', type=str, default='testA', help='real2avatar')
parser.add_argument('--YtoX', type=str, default='testB', help='avatar2real')
# 输入测试数据集jpg：/home/ma-user/modelarts/inputs/data_url_0/avatar_data/testA
parser.add_argument('--datasets_dir', type=str,
                    default='/home/ma-user/modelarts/inputs/data_url_0/avatar_data',
                    help='Input datasets path.')
# 输出bin：/home/ma-user/modelarts/outputs/train_url_0/bin/testA
parser.add_argument('--output_dir', type=str,
                    default='/home/ma-user/modelarts/outputs/train_url_0/bin',
                    help='Output dataset path.')
FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir+'/'+FLAGS.XtoY)
    os.makedirs(FLAGS.output_dir+'/'+FLAGS.YtoX)

def parse_dir(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            pic_path = input_dir + "/" + file
            print("start to process %s" % pic_path)

            # image = tf.io.read_file(filename=pic_path)
            # image = tf.image.decode_jpeg(image, channels=1)
            # image = tf.image.resize_images(image, size=(256, 256))
            # image = convert2float(image)
            # image.set_shape([256, 256, 1])
            #
            # with tf.Session() as sess:
            #     image_numpy = image.eval()

            image = load_test_data(image_path=pic_path)  # type(numpy.ndarray)
            file = file[:-4]
            # save the pic as .bin format for Ascend310 infer.
            image.tofile(output_dir + "/" + file + ".bin")


def main(_):
    print('Transfer datasets for XtoY model...')
    parse_dir(FLAGS.datasets_dir+"/"+FLAGS.XtoY, FLAGS.output_dir+"/"+FLAGS.XtoY)
    print('Transfer datasets for YtoX model...')
    parse_dir(FLAGS.datasets_dir+"/"+FLAGS.YtoX, FLAGS.output_dir+"/"+FLAGS.YtoX)


if __name__ == '__main__':
    tf.app.run()
