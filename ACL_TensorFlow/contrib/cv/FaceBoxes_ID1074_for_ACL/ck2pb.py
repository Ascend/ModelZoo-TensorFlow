# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os, sys
import argparse
from src import Detector, AnchorGenerator, FeatureExtractor,Evaluator

base_path=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/../")

# from nets.deep_sort.network_definition import create_network


def main():

    tf.reset_default_graph()

    # set inputs node
    inputs = tf.placeholder(dtype=tf.uint8,shape=[1,1024,1024,3], name="image_tensor")
    pics=tf.cast(inputs,tf.float32)
    pics=tf.divide(pics,255)


    feature_extractor = FeatureExtractor(False)

    # anchor maker
    anchor_generator = AnchorGenerator()

    # add box/label predictors to the feature extractor
    detector = Detector(pics, feature_extractor, anchor_generator)

    boxes, scores, num_boxes = detector.get_predictions(
        score_threshold=0.05,
        iou_threshold=0.3,
        max_boxes=200
    )

    boxes_=tf.identity(boxes, name="boxes")
    scores=tf.identity(scores, name="scores")
    num_boxes_=tf.identity(num_boxes, name="num_boxes")

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph="faceboxes.pb"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, "./models/run00/model.ckpt-300000")
        output_nodes=['nms/map/TensorArrayStack/TensorArrayGatherV3','nms/map/TensorArrayStack_1/TensorArrayGatherV3','nms/map/TensorArrayStack_2/TensorArrayGatherV3']
        # output_nodes=['boxes','scores','num_boxes']
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_nodes)

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")

if __name__ == '__main__':

    main()

