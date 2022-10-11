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
the prerequisite is that there are the files converted by tool named msame using om module
This is the file for get the index of the max probability.
"""
import os
import json
import numpy as np
import tensorflow as tf
import absl.flags as flags
import absl.app as app
import cv2
from inception_preprocessing import preprocess_for_eval
from tensorflow.python.platform import gfile
FLAGS = flags.FLAGS
flags.DEFINE_string("result_image_dir",
                    "log/result_image/20221011_12_21_54_254035",
                    "")
flags.DEFINE_string("original_jpeg_image", "log/original_jpeg_image", "the file store the JPEG file")
flags.DEFINE_integer("start", 1, "")
flags.DEFINE_integer("end", 100, "")
label_dict = dict()


def predict():
    global label_dict
    tf.reset_default_graph()
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    with gfile.FastGFile(
            "./log/frozen_pb_file/frozen_model.pb",
            'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    for i in range(FLAGS.start, FLAGS.end+1, 1):
        filename = "ILSVRC2012_val_" + str(i).zfill(8) + ".JPEG"
        filepath = os.path.join(FLAGS.original_jpeg_image, filename)
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.uint8)
        processed_img = preprocess_for_eval(img, 224, 224)
        processed_img = tf.expand_dims(processed_img, 0)
        processed_img = sess.run(processed_img)
        input_x = sess.graph.get_tensor_by_name('input:0')
        op = sess.graph.get_tensor_by_name('module/Squeeze_1:0')
        res = sess.run(op, feed_dict={input_x: processed_img})
        label_index = np.argmax(res)
        label_dict.setdefault(filename, label_index)

def offline_reasoning():
    global label_dict
    sum = 0
    hit = 0
    dirname = FLAGS.result_image_dir
    if FLAGS.end > len(os.listdir(FLAGS.original_jpeg_image)):
        raise "the end index Exceeded the maximum boundary."
    for i in range(FLAGS.start, FLAGS.end+1, 1):
        sum += 1
        filename = "ILSVRC2012_val_" + str(i).zfill(8) + ".JPEG"
        true_label = label_dict[filename]
        filename = filename + "_output_0.bin"
        filepath = os.path.join(dirname, filename)
        prob_distribution = np.fromfile(filepath, dtype='float32')
        inf_label = int(np.argmax(prob_distribution))
        if int(inf_label) == int(true_label):
            hit += 1
        print(f"image filename is {filename} , om_inference_label is {inf_label}, pb_inference_label {true_label}")
    print(f"the sum is {sum}, hit {hit}, the percentage of hits is {hit * 100 / sum}%")

def main(unused_argv):
    predict()
    offline_reasoning()

if __name__ == "__main__":
    app.run(main)
