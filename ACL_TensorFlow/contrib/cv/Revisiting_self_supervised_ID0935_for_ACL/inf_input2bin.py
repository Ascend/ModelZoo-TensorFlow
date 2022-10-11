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
"""
This is the file for convert image to the type of bin.
"""
import os
import cv2
import numpy as np
from inception_preprocessing import preprocess_for_eval
from npu_bridge.npu_init import *
import tensorflow as tf
import absl.flags as flags
import absl.app as app
import absl.logging as logging
FLAGS = flags.FLAGS
flags.DEFINE_integer("start", 1, "from where to start convert the image")
flags.DEFINE_integer("end", 3, "from where to end to convert the image")
flags.DEFINE_string("original_jpeg_image", "./log/original_jpeg_image", "")
flags.DEFINE_string("bin_image", "./log/bin_image", "")

def convert2bin():
    for i in range(FLAGS.start, FLAGS.end+1, 1):
        filename = "ILSVRC2012_val_"+str(i).zfill(8)+".JPEG"
        filepath = os.path.join(FLAGS.original_jpeg_image, filename)
        print(f"start to process {filepath}")
        img_org = cv2.imread(filepath)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(value=img_org, dtype=tf.uint8, name="input")
        img = preprocess_for_eval(img, 224, 224)
        with tf.Session() as sess:
            res = img.eval()
        res.tofile(FLAGS.bin_image + "/" + filename + ".bin")

def main(unused_argv):
    logging.info("-----> start convert the raw image to the bin type-----<")
    convert2bin()
    logging.info("-----> convert success -----<")

if __name__ == "__main__":
    app.run(main)