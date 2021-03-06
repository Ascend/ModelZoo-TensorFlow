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

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
import xml.etree.ElementTree as ET
from utils.utils_bbox import DecodeBox
from utils.utils_map import  get_map
from PIL import Image

MINOVERLAP = 0.5
classes_path = 'model_data/voc_classes.txt'
output_path = 'out/2022410_0_32_31_672278'
VOCdevkit_path = 'VOCdevkit'
map_out_path    = 'map_out_om'
anchors_path = 'model_data/yolo_anchors.txt'
anchors, num_anchors = get_anchors(anchors_path)
class_names, num_classes = get_classes(classes_path)


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   ???y??????????????????????????????????????????????????????????????????
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   ??????????????????offset????????????????????????????????????????????????????????????
        #   new_shape???????????????????????????
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ------------------------------------------#
    #   grid_shape??????????????????????????????
    # ------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    # --------------------------------------------------------------------#
    #   ????????????????????????????????????????????????shape???(20, 20, num_anchors, 2)
    # --------------------------------------------------------------------#
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    # ---------------------------------------------------------------#
    #   ????????????????????????????????????shape???(20, 20, num_anchors, 2)
    # ---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # ---------------------------------------------------#
    #   ????????????????????????(batch_size, 20, 20, 3, 85)
    #   85????????????4 + 1 + 80
    #   4???????????????????????????????????????
    #   1???????????????????????????
    #   80??????????????????????????????
    # ---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # ------------------------------------------#
    #   ?????????????????????????????????????????????
    # ------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) * 2 - 0.5 + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = (K.sigmoid(feats[..., 2:4]) * 2) ** 2 * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    # ------------------------------------------#
    #   ???????????????????????????
    # ------------------------------------------#
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   ?????????loss???????????????grid, feats, box_xy, box_wh
    #   ????????????????????????box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def Decodebox(outputs,
            anchors,
            num_classes,
            image_shape,
            input_shape,
            #-----------------------------------------------------------#
            #   13x13?????????????????????anchor???[116,90],[156,198],[373,326]
            #   26x26?????????????????????anchor???[30,61],[62,45],[59,119]
            #   52x52?????????????????????anchor???[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):

    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    for i in range(len(outputs)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)

    #------------------------------------------------------------------------------------------------------------#
    #   ???????????????????????????????????????letterbox_image?????????????????????????????????????????????box_xy, box_wh?????????????????????????????????
    #   ????????????????????????????????????????????????????????? ???box_xy??????box_wh?????????y_min,y_max,xmin,xmax
    #   ??????????????????letterbox_image???????????????????????????box_xy, box_wh?????????????????????????????????
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   ????????????????????????score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []

    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   ????????????box_scores >= score_threshold??????????????????
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   ???????????????
        #   ???????????????????????????????????????
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   ?????????????????????????????????
        #   ??????????????????????????????????????????????????????
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


# def generate(model_path, anchors_mask, num_classes, phi, output):
#     model_path = os.path.expanduser(model_path)
#     assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
#     #
#     # yolo_model = yolo_body([None, None, 3], anchors_mask, num_classes, phi)
#     # yolo_model.load_weights(model_path)
#     # print('{} model, anchors, and classes loaded.'.format(model_path))
#     #
#     # # anchors, num_anchors = get_anchors(anchors_path)
#     # # class_names, num_classes = get_classes(classes_path)
#     # # ---------------------------------------------------------#
#     # #   ???yolo_eval???????????????????????????????????????????????????
#     # #   ?????????????????????????????????????????????????????????????????????
#     # # ---------------------------------------------------------#
#     boxes, scores, classes = Decodebox(
#         outputs=output,
#         anchors=anchors,
#         num_classes=num_classes,
#         image_shape=K.placeholder(shape=(2, )),
#         input_shape=[640, 640],
#         anchor_mask=anchors_mask,
#         max_boxes=100,
#         confidence=0.5,
#         nms_iou=0.3,
#         letterbox_image=True
#     )
#     return boxes, scores, classes

def main():
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))



    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        image = cvtColor(image)
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        feats = []
        feats_path0 = os.path.join(output_path, "image_" + image_id + "_output_0.txt")
        feats_path1 = os.path.join(output_path, "image_" + image_id + "_output_1.txt")
        feats_path2 = os.path.join(output_path, "image_" + image_id + "_output_2.txt")
        feats0 = np.loadtxt(feats_path0)
        feats0 = np.reshape(feats0, newshape=(1, 80, 80, 75))
        feats0 = feats0.astype("float32")
        feats0 = tf.convert_to_tensor(feats0)
        feats.append(feats0)

        feats1 = np.loadtxt(feats_path1)
        feats1 = np.reshape(feats1, newshape=(1, 40, 40, 75))
        feats1 = feats1.astype("float32")
        feats1 = tf.convert_to_tensor(feats1)
        feats.append(feats1)

        feats2 = np.loadtxt(feats_path2)
        feats2 = np.reshape(feats2, newshape=(1, 20, 20, 75))
        feats2 = feats2.astype("float32")
        feats2 = tf.convert_to_tensor(feats2)
        feats.append(feats2)

        out_boxes, out_scores, out_classes = Decodebox(outputs=feats,
                                                        anchors=anchors,
                                                        num_classes=num_classes,
                                                        image_shape=[image.size[1], image.size[0]],
                                                        input_shape=[640, 640],
                                                        # anchor_mask=anchors_mask,
                                                        # max_boxes=100,
                                                        # confidence=0.5,
                                                        # nms_iou=0.3,
                                                        # letterbox_image=True
                                                        )
        out_boxes = K.eval(out_boxes)
        out_scores = K.eval(out_scores)
        out_classes = K.eval(out_classes)

        # with tf.Session() as sess:
        #     out_boxes = out_boxes.eval(session=sess, feed_dict={out_boxes: zero_array1})
        #     out_scores = out_scores.eval(session=sess, feed_dict={out_scores: zero_array2})
        #     out_classes = out_classes.eval(session=sess, feed_dict={out_classes: zero_array3})

        for i, c in enumerate(out_classes):
            predicted_class = class_names[int(c)]
            score = str(out_scores[i])
            top, left, bottom, right = out_boxes[i]
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        print("Get ground truth result.")

    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    print("Get map.")
    get_map(MINOVERLAP, True, path=map_out_path)
    print("Get map done.")

if __name__=="__main__":
    main()