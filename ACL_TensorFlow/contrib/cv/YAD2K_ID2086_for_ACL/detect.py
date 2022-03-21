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
import numpy as np
import keras.backend as K

from nets.yololoss import yolo_head
from data_process.config import anchors,classes,input_shape,batch_size
from nets.v2net import model_body
import cv2
from PIL import Image



# 过滤掉得分比较低的预测
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    # （1,13,13,5,1）  X  （1,13,13,5,47 ）-->  （1,13,13,5,47）
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)  # (1,13,13,5) 获得最大的类别的索引
    box_class_scores = K.max(box_scores, axis=-1)  # （1,13,13,5）获得得分的最大值
    prediction_mask = box_class_scores >= threshold  # （1,13,13,5） 目标mask,过滤掉那些得分比较低的预测

    boxes = tf.boolean_mask(boxes, prediction_mask)  # 保留得分比较高的那些boxes
    scores = tf.boolean_mask(box_class_scores, prediction_mask)  # 保留得分比较高的那些 scores
    classes = tf.boolean_mask(box_classes, prediction_mask)  # 保留得分比较高的那些 类别
    return boxes, scores, classes


# 转化为左上角右下角坐标
def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    # x_min，y_min，x_max，y_max
    return K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                          box_maxes[..., 0:1], box_maxes[..., 1:2]])


# 对预测结果进行过滤及非极大值抑制
def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    # (batch,13,13,5,2),(batch,13,13,5,2),(batch,13,13,5,1),(batch,13,13,5,47)
    print("我正在执行yolo_eval")
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs  # yolo的输出
    boxes = yolo_boxes_to_corners(box_xy, box_wh)  # 转化为左上角、右下角的形式
    # 过滤掉得分比较低的预测
    boxes, scores, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)

    height = image_shape[0]  # 输入图片的高/宽
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    image_dims = K.cast(image_dims, K.dtype(boxes))
    boxes = boxes * image_dims  # 缩放到输入图片的大小

    # 非极大值抑制
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    boxes = K.gather(boxes, nms_index)  # 得到进行非极大值抑制后的保留boxes
    scores = K.gather(scores, nms_index)  # 得到进行非极大值抑制后的保留 scores
    classes = K.gather(classes, nms_index)  # 得到进行非极大值抑制后的保留 classes
    print("box",type(boxes))
    return boxes, scores, classes

if __name__ == "__main__":

    CUDA_VISIBLE_DEVICES = 0,3
    #加载网络模型
    model_path = r"./logs/ep081-loss0.495.h5"
    test_path = r"./test_img/000005.jpg"
    model_body.load_weights(model_path) #加载权重

    num_classes = len(classes)
    #print(num_classes)
    num_anchors = len(anchors)
    # 加载图片
    image = cv2.imread(test_path)
    origin_img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，转换为RGB
    image = cv2.resize(image, input_shape)
    image = image / 255
    image = np.expand_dims(image, 0)

    #y = model_body.predict(image)  # 检测图片
    yolo_outputs = yolo_head(model_body.output, anchors, num_classes)  # 对网络输出进行编码
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, pred_classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.15, iou_threshold=0.4)
    sess = K.get_session()
    origin_shape = origin_img.shape[0:2]
    origin_imgs = cv2.resize(origin_img, input_shape)

    boxes, scores, pred_classes = sess.run(
                [boxes, scores, pred_classes],
                feed_dict={
                    model_body.input: image,
                    input_image_shape: [input_shape[1], input_shape[0]],
                    K.learning_phase(): 0
                })
    #
    # 绘制预测结果
    for i, (box,score) in enumerate(zip(boxes,scores)):
        cv2.rectangle(origin_imgs, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
        cv2.putText(origin_imgs, classes[pred_classes[i]]+" "+str(round(score,2)), (int(box[0]), int(box[1])), 1, 1, (0, 0, 255), 1)

    # 缩放到原图并显示
    tmp = cv2.resize(origin_imgs, (origin_shape[1], origin_shape[0]))
    tmp = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
    tmp.show()