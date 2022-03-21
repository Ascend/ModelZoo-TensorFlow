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


# Encode the network output
def yolo_head(feats, anchors, clsnum):

    feats = tf.convert_to_tensor(feats, dtype=tf.float32)  # 将numpy数组转换为tf张量，
    num_anchors = len(anchors)  # 5
    # change to shape:(batch, height, width, 5, 2)
    # The various methods used in YAD2K have the same effect
    # anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    anchors_tensor = K.reshape(tf.convert_to_tensor(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    conv_dims = K.shape(feats)[1:3]  # Get the first and second dimensions of the feature map ---> [13, 13] ,shape: (2,)
    conv_height_index = K.arange(0, stop=conv_dims[0])  # Get a one-dimensional array of [0--12], representing high
    conv_width_index = K.arange(0, stop=conv_dims[1])  # Get a one-dimensional array of [0--12], representing width

    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])  # Perform one-dimensional tiling, [0--12 0--12] repeat 13 times shape(13*13,)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0),
                              [conv_dims[0], 1])  # [[0--12],[0--12]] shape:（13,13）
    # [0*13, 1*13 .... 12*13]-->  [0...0,1...1,2...2,....,12...12]
    conv_width_index = K.flatten(K.transpose(conv_width_index))  # shape(13*13，)

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))  # (2,13*13)---transpose--->(13*13,2)
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])  # (1, 13, 13, 1, 2）
    conv_index = K.cast(conv_index, K.dtype(feats))  # Data type conversion

    # feats：（batch,13,13,260）--->(batch,13,13,5,(20+5))
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, clsnum + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))  # (1, 1, 1, 1, 2)

    box_xy = K.sigmoid(feats[..., :2])  # Network output, xy encoding
    box_wh = K.exp(feats[..., 2:4])  # Network output, wh encoding
    box_confidence = K.sigmoid(feats[..., 4:5])  # Confidence coding
    box_class_probs = K.softmax(feats[..., 5:])  # Category prediction output

    box_xy = (box_xy + conv_index) / conv_dims  # The network output is normalized with respect to the feature map after adding the position of the grid
    box_wh = box_wh * anchors_tensor / conv_dims
    # (batch,13,13,5,2),(batch,13,13,5,2),(batch,13,13,5,1),(batch,13,13,5,20)
    return box_xy, box_wh, box_confidence, box_class_probs


# Define loss function
def yolo_loss(args,
              anchors,
              num_classes):

    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args  # As above
    num_anchors = len(anchors)  # 5
    object_scale = 5  # Calculate the correction coefficient of the confidence when the object is in the grid cell
    no_object_scale = 1  # Calculate the correction coefficient of the confidence when the object is in the grid cell
    class_scale = 1  # Calculate the correction factor of the classification loss
    coordinates_scale = 1  # Calculate the correction factor for coordinate loss

    # Get the encoding of the network output
    #
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(yolo_output, anchors, num_classes)

    yolo_output_shape = K.shape(yolo_output)  # （batch,13,13,125）
    feats = K.reshape(yolo_output, [-1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
                                    num_classes + 5])  # （batch,13,13,5,25）
    # The x, y, w, h of pred_boxes are combined to calculate the coordinate loss with matching_true_boxes,
    # shape: [batch,13,13,5,4]
    pred_boxes = K.concatenate((K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)  # (batch,13,13,5,4）

    pred_xy = K.expand_dims(pred_xy, 4)  # (batch,13,13,5,2）-->(batch,13,13,5,1,2）
    pred_wh = K.expand_dims(pred_wh, 4)  # (batch,13,13,5,2）-->(batch,13,13,5,1,2）

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half  # Coordinates of the upper left corner
    pred_maxes = pred_xy + pred_wh_half  # Coordinates of the upper right corner

    true_boxes_shape = K.shape(true_boxes)
    # (Number of pictures, several actual boxes, [x,y,w,h,class])
    # three-dimensional, x,y,w,h are all (0,1) relative to the grid cell

    # (batch,1,1,1,图片有几个框,5)
    # shape:[batch,1,1,1,-1,5],batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]])

    true_xy = true_boxes[..., 0:2]  # (batch,1,1,1,box numbers,2）
    true_wh = true_boxes[..., 2:4]  # (batch,1,1,1,box numbers,2）

    # 'Calculate the coordinates of the upper left vertex and the lower right vertex of true_boxes'
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # The intersection area between the prediction box and the ground truth box (b,13,13,5,20)

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # Prediction box area(b,13,13,5,1)
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # Real box area(b,1,1,1,20)

    union_areas = pred_areas + true_areas - intersect_areas  # iou (b,13,13,5,20) IOU of each prediction box and all ground truth boxes
    iou_scores = intersect_areas / union_areas  # compute all anchor boxes and true_boxes  IOU,shape:[-1,13,13,5,2,1]

    # It’s very interesting here. If two true_boxes fall in the same grid cell, I only take the one with the largest iou,
    # Because the value of best_iou only cares about the largest iou in this grid cell, not which true_boxes it comes from.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores. (b,13,13,5)
    best_ious = K.expand_dims(best_ious)  # (b,13,13,5,1)  'shape:[1,-1,13,13,5,1]'

    #'Select the IOU greater than 0.6, do not pay attention to its loss. cast() function,
    # the first parameter is bool value, dtype is int, it will be converted to 0,1'
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))  # (b,13,13,5,1),Keep those prediction boxes with larger iou

    # (b,13,13,5,1)
    no_object_weights = (no_object_scale * (1 - object_detections) * (1 - detectors_mask))  # Anchor without target box
    no_objects_loss = no_object_weights * K.square(-pred_confidence)  # loss1  Loss without target box

    objects_loss = (object_scale * detectors_mask * K.square(1 - pred_confidence))  # loss2 Loss with target box
    # compute confidence_loss，
    # no_objects_loss is the error of calculating the background,
    # Objects_loss is to calculate the error of anchor_boxes matching true_box.
    # Compared with no_objects_loss, it pays more attention to this part of the error, and its correction factor is 5.
    confidence_loss = objects_loss + no_objects_loss  # Confidence loss

    # Classification loss
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')  # Which anchor contains the target
    matching_classes = K.one_hot(matching_classes, num_classes)  # one-hot  (b,13,13,5,20)
    # 'Calculate the classification_loss, the difference of the 20-dimensional vector'
    classification_loss = (
            class_scale * detectors_mask * K.square(matching_classes - pred_class_prob))  # Classification loss (b,13,13,5,20)

    # coordinates_loss
    # Calculate coordinates_loss, x, y are the mean square loss of the offset,
    # w, h are the mean square loss of the logarithm,
    # which is similar to the mean square of the square root difference in YOLOv1, and the effect is slightly better.
    matching_boxes = matching_true_boxes[..., 0:4]  # (b,13,13,5,4)
    coordinates_loss = (coordinates_scale * detectors_mask * K.square(matching_boxes - pred_boxes))  # coordinates_loss

    # Sum all to one value
    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)  # total loss

    # 打印loss
    if False:
        tf.print("yolo_loss", total_loss, {"conf_loss": confidence_loss_sum},
                 {"class_loss": classification_loss_sum},
                 {"box_coord_loss": coordinates_loss_sum}, output_stream=sys.stdout)

    return total_loss

# # loss layer
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
#         loss = yolo_loss(inputs, self.anchors, self.num_classes)  # loss
#         self.add_loss(loss, inputs=True)  # record loss
#         self.add_metric(loss, name="yolo_loss")  # estimate this layer
#         return loss
