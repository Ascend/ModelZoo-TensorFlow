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
from npu_bridge.npu_init import *

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k
from nets.yololoss import yolo_head
from data_process.config import anchors, classes, input_shape, batch_size, colors
from nets.v2net import model_body
import cv2
from PIL import Image
import argparse
from data_process import tool


# filter the
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    # (1,13,13,5,1)  X  (1,13,13,5,20)-->  (1,13,13,5,20)
    box_scores = box_confidence * box_class_probs
    box_classes = k.argmax(box_scores, axis=-1)  # (1,13,13,5)
    box_class_scores = k.max(box_scores, axis=-1)  # (1,13,13,5)

    prediction_mask = box_class_scores >= threshold  # (1,13,13,5)

    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes



def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    # x_min，y_min，x_max，y_max
    return k.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                          box_maxes[..., 0:1], box_maxes[..., 1:2]])


# yuce result nms
def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    # (batch,13,13,5,2),(batch,13,13,5,2),(batch,13,13,5,1),(batch,13,13,5,20)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs  # yolo_head output
    boxes = yolo_boxes_to_corners(box_xy, box_wh)  # change to left corner and right corner
    # drop the low score
    boxes, scores, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)

    height = image_shape[0]  # 'image_shape,(416x416)'
    width = image_shape[1]
    image_dims = k.stack([height, width, height, width])
    image_dims = k.reshape(image_dims, [1, 4])
    image_dims = k.cast(image_dims, k.dtype(boxes))
    boxes = boxes * image_dims  # gain really x,y,w,h'

    # iou_threshold defalut 0.5
    max_boxes_tensor = k.constant(max_boxes, dtype='int32')
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    boxes = k.gather(boxes, nms_index)  # last boxes
    scores = k.gather(scores, nms_index)  # last scores
    classes = k.gather(classes, nms_index)  # last classes
    return boxes, scores, classes


if __name__ == "__main__":

    print("------begin the file------")
    parser = argparse.ArgumentParser()
    # bos first
    parser.add_argument("--dataset", type=str, default="./output")  ## config.modelarts_data_dir  namely /cache/DataSet
    parser.add_argument("--result", type=str, default="./dataset")  ## config.modelarts_result_dir namely  /cache/result

    # for run over
    parser.add_argument("--obs_dir", type=str)  ## last output fileaddress s3://yolov2/yolov2forfen/output/V0027/

    config1 = parser.parse_args()
    print("config1.dataset", config1.dataset)  ## config.modelarts_data_dir  namely  /cache/DataSet
    print("config1.result", config1.result)  ## config.modelarts_result_dir namely  /cache/result

    # for run over
    print("config1.obs_dir", config1.obs_dir)  ## last output fileaddress s3://yolov2/yolov2forfen/output/V0027/

    # HUA WEI
    print("-----HuaWei------")
    # Create session
    config_session = tf.ConfigProto()
    custom_op = config_session.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config_session.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Must be explicitly closed
    config_session.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # Must be explicitly closed
    sess = tf.Session(config=config_session)

    # Load the network model
    model_inobs = config1.dataset + "/ep099-loss1.315.h5"
    model_path = model_inobs

    # Load the network weights
    model_body.load_weights(model_path)
    # Read images in a loop
    filepic_path = config1.dataset + '/test_img'
    image_names = tool.find_files(filepic_path)
    print("image_names =   ", image_names)

    for filename in image_names:
        filename1 = filename[24:]
        print("filename1 =   ", filename1)
        # test_path = config1.dataset+'/test_img/sample_office.jpg'

        num_classes = len(classes)
        # print(num_classes)
        num_anchors = len(anchors)
        # Load iamges
        image = cv2.imread(filename)
        origin_img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv BGR change to RGB
        image = cv2.resize(image, input_shape)
        image = image / 255
        image = np.expand_dims(image, 0)

        # y = model_body.predict(image)
        yolo_outputs = yolo_head(model_body.output, anchors, num_classes)  # Encode network output
        input_image_shape = k.placeholder(shape=(2,))
        boxes, scores, pred_classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.15,
                                                iou_threshold=0.4)
        sess = k.get_session()
        origin_shape = origin_img.shape[0:2]
        origin_imgs = cv2.resize(origin_img, input_shape)

        boxes, scores, pred_classes = sess.run(
            [boxes, scores, pred_classes],
            feed_dict={
                model_body.input: image,
                input_image_shape: [input_shape[1], input_shape[0]],
                k.learning_phase(): 0
            })
        #
        # Plot the forecast results
        # Length and width of the original image
        image_h = origin_imgs.shape[0]
        image_w = origin_imgs.shape[1]
        thick = int((image_h + image_w) // 300)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # cv2.rectangle(img, (x1, y1)left up corner, (x2, y2)right down corner, (255,0,0)color, 2 box bold )
            cv2.rectangle(origin_imgs, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[pred_classes[i]],
                          thick)
            # cv2.putText(img, str(i), (123,456)left up corner, font, 2 font size, (0,255,0)color, 3)
            cv2.putText(origin_imgs, classes[pred_classes[i]] + " " + str(round(score, 2)),
                        (int(box[0]) - 8, int(box[1]) - 8), 0, 1e-3 * image_h, colors[pred_classes[i]], thick // 2)

        # Zoom to the original image and display
        tmp = cv2.resize(origin_imgs, (origin_shape[1], origin_shape[0]))
        # array change into image
        tmp = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))

        # tmp.save(config1.result+"/result_test.jpg")  ##/cache/result
        savepath = "/result_" + filename1
        tmp.save(config1.result + savepath)

    from help_modelarts import modelarts_result2obs

    print("------Get result-------")
    config1.profiling = False
    modelarts_result2obs(config1)

    sess.close()
