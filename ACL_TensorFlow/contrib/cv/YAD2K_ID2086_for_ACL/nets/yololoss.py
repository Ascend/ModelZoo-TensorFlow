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

from keras.layers import Layer
import keras.backend as K
from keras import layers
import tensorflow as tf
from nets.v2net import model_body
# 对网络输出进行编码
def yolo_head(feats, anchors, clsnum):
    """feats 网络输出（batch,13,13,125）
       anchors:(5,2)
       clsnum:20
    """
    print("我正在执行 yolo_head")
    print("feats",feats)
    feats = tf.convert_to_tensor(feats, dtype=tf.float32)  # 将numpy数组转换为tf张量，
    num_anchors = len(anchors)  # 5
    # 转化为：shape:（ batch, height, width, 5, 2）
    anchors_tensor = K.reshape(tf.convert_to_tensor(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    conv_dims = K.shape(feats)[1:3]  # 得到特征图的第1,2个维度---> [13, 13] ,shape：（2，）
    conv_height_index = K.arange(0, stop=conv_dims[0])  # 得到[0--12]的一维数组，代表高
    conv_width_index = K.arange(0, stop=conv_dims[1])  # 得到[0--12]的一维数组，代表宽

    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])  # 进行一维平铺，[0--12 0--12] 重复13次 shape(13*13，)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0),
                              [conv_dims[0], 1])  # [[0--12],[0--12]] shape:（13,13）
    # [0*13, 1*13 .... 12*13]-->  [0...0,1...1,2...2,....,12...12]
    conv_width_index = K.flatten(K.transpose(conv_width_index))  # shape(13*13，)

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))  # (2,13*13)---transpose--->(13*13,2)
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])  # (1, 13, 13, 1, 2）
    conv_index = K.cast(conv_index, K.dtype(feats))  # 数据类型转化

    # feats：（batch,13,13,125）--->(batch,13,13,5,(20+5))
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, clsnum + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))  # (1, 1, 1, 1, 2)

    box_xy = K.sigmoid(feats[..., :2])  # 网络输出，xy编码
    box_wh = K.exp(feats[..., 2:4])  # 网络输出，wh编码
    box_confidence = K.sigmoid(feats[..., 4:5])  # 置信度编码
    box_class_probs = K.softmax(feats[..., 5:])  # 类别预测输出

    box_xy = (box_xy + conv_index) / conv_dims  # 网络输出加上gride的位置之后相对于特征图进行归一化
    box_wh = box_wh * anchors_tensor / conv_dims
    # (batch,13,13,5,2),(batch,13,13,5,2),(batch,13,13,5,1),(batch,13,13,5,20)
    return box_xy, box_wh, box_confidence, box_class_probs


# 定义loss函数
def yolo_loss(args,
              anchors,
              num_classes):
    """args 包含如下4个：
    yolo_output (batch, 13, 13, 125)
    true_boxes：[x_center, y_center, width, height, class]
    detectors_mask: shape: [13, 13, 5, 1],表示由哪个anchor负责预测目前
    matching_true_boxes: shape 同上 ，编码后的真实label,相对于特征图的

    anchors：anchor：shape(5,2)
    num_classes:类别个数 20
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args  # 如上
    num_anchors = len(anchors)  # 5
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1

    # 得到网络输出的编码
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(yolo_output, anchors, num_classes)

    yolo_output_shape = K.shape(yolo_output)  # （batch,13,13,260）
    feats = K.reshape(yolo_output, [-1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
                                    num_classes + 5])  # （batch,13,13,5,52）
    pred_boxes = K.concatenate((K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)  # (batch,13,13,5,4）

    pred_xy = K.expand_dims(pred_xy, 4)  # (batch,13,13,5,2）-->(batch,13,13,5,1,2）
    pred_wh = K.expand_dims(pred_wh, 4)  # 同上

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half  # 左上角坐标
    pred_maxes = pred_xy + pred_wh_half  # 右下角坐标

    true_boxes_shape = K.shape(true_boxes)

    # （batch,1,1,1,20,5）
    true_boxes = K.reshape(true_boxes, [true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]])

    true_xy = true_boxes[..., 0:2]  # (batch,1,1,1,20,2）
    true_wh = true_boxes[..., 2:4]  # (batch,1,1,1,20,2）

    # 找到每个预测box与每个真实label的 IOU
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # 预测框与真实框的相交区域 (b,13,13,5,20)
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # 预测框的面积(b,13,13,5,1)
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # 真实框的面积(b,1,1,1,20)
    union_areas = pred_areas + true_areas - intersect_areas  # iou (b,13,13,5,20) 每个预测框与所有的真实框的  区域并集
    iou_scores = intersect_areas / union_areas  # iou

    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores. (b,13,13,5)
    best_ious = K.expand_dims(best_ious)  # (b,13,13,5,1)
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))  # (b,13,13,5,1),保留iou比较大的那些预测框

    # (b,13,13,5,1)
    no_object_weights = (no_object_scale * (1 - object_detections) * (1 - detectors_mask))  # 没有目标框的anchor
    no_objects_loss = no_object_weights * K.square(-pred_confidence)  # loss1  没有目标框的loss
    objects_loss = (object_scale * detectors_mask * K.square(1 - pred_confidence))  # loss2 有目标框的loss
    confidence_loss = objects_loss + no_objects_loss  # 置信度loss

    # 分类损失
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')  # 哪个anchor 包含目标
    matching_classes = K.one_hot(matching_classes, num_classes)  # one-hot  (b,13,13,5,47)
    classification_loss = (
                class_scale * detectors_mask * K.square(matching_classes - pred_class_prob))  # 分类损失 (b,13,13,5,47)

    # 坐标损失
    matching_boxes = matching_true_boxes[..., 0:4]  # (b,13,13,5,4)
    coordinates_loss = (coordinates_scale * detectors_mask * K.square(matching_boxes - pred_boxes))  # 坐标损失

    # 全部求和为一个值
    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)  # 总loss

    # 打印loss
    if False:
        tf.print("yolo_loss", total_loss, {"conf_loss": confidence_loss_sum},
                 {"class_loss": classification_loss_sum},
                 {"box_coord_loss": coordinates_loss_sum}, output_stream=sys.stdout)

    return total_loss


# # 封装一个loss layer
# class YoloLoss(Layer):
#     def __init__(self, anchors, num_classes, **kwargs):
#         super(YoloLoss, self).__init__(**kwargs)
#         self.anchors = anchors
#         self.num_classes = num_classes
#         self._name = "yolo_loss"
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0],)
#
#     def call(self, inputs, **kwargs):
#         loss = yolo_loss(inputs, self.anchors, self.num_classes)  # 计算loss
#         self.add_loss(loss, inputs=True)  # 记录loss
#         self.add_metric(loss, name="yolo_loss")  # 加入评估到当前layer
#         return loss
