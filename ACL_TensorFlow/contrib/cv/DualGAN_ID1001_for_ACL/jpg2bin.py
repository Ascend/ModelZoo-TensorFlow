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

python3 jpg2bin.py --datasets_dir " ./datasets/" --output_dir "./output/"

"""
import os
import argparse
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default='', help="Input datasets path.")
    parser.add_argument("--output_dir", type=str, default='', help="Output dataset path.")
    config = parser.parse_args()

    if os.path.exists(config.output_dir):
        pass
    else:
        os.makedirs(config.output_dir)

    for file in os.listdir(config.datasets_dir):
        if file.endswith('.jpg'):
            pic_path = config.datasets_dir + "/" + file
            print("start to process %s" % pic_path)

            image = tf.io.read_file(filename=pic_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize_images(image, size=(256, 256))
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = (image / 127.5) - 1.0
            image.set_shape([256, 256, 3])

            with tf.Session() as sess:
                image_numpy = image.eval()
            # save the pic as .bin format for Ascend310 infer.
            image_numpy.tofile(config.output_dir + "/" + file + ".bin")
