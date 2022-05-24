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
import tensorflow as tf
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import os
import argparse
from utils import session_config, check_dir
import numpy as np
from generate_fake_img import generate_img_of_one_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # platform arguments (Huawei Ascend)
    parser.add_argument("--chip", type=str, default="gpu", help="run on which chip, cpu or gpu or npu")
    # data arguments
    parser.add_argument("--output", type=str, default="../output", help="output path")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_h", type=int, default=32, help="image height")
    parser.add_argument("--img_w", type=int, default=32, help="image width")
    parser.add_argument("--train_img_size", type=int, default=32,
                        help="image will be resized to this size when training")
    # model arguments
    parser.add_argument("--base_channel", type=int, default=96, help="base channel number for G and D")
    parser.add_argument("--z_dim", type=int, default=120, help="latent space dimensionality")
    parser.add_argument("--truncation", type=float, default=2.0, help="truncation threshold")
    parser.add_argument("--ema", type=bool, default=True, help="use exponential moving average for G")
    parser.add_argument("--shared_dim", type=int, default=128, help="shared embedding dimensionality")
    args = parser.parse_args()

    # get output dir
    inference_path = os.path.join(args.output, "inference", str(args.train_img_size))
    check_dir(inference_path)
    # pb path
    pb_path = os.path.join(args.output, "pb_model", str(args.train_img_size))
    graph_pb_path = os.path.join(pb_path, "tmp_model.pb")
    model_pb_path = os.path.join(pb_path, "model.pb")
    final_pb_path = os.path.join(pb_path, "final_model.pb")

    tf.reset_default_graph()
    with tf.gfile.FastGFile(final_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # text_format.Merge(f.read(), graph_def)

        _ = tf.import_graph_def(graph_def, name="")

    config = session_config(args)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        z = sess.graph.get_tensor_by_name("z:0")
        y = sess.graph.get_tensor_by_name("y:0")
        fake_img = sess.graph.get_tensor_by_name("output:0")

        class_labels = np.random.randint(0, 11, size=(args.batch_size, 1))
        generate_img_of_one_class(args, class_labels, "inference.jpg", inference_path, sess, fake_img, z, y)
