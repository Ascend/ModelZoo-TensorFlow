#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import StreamManagerApi.py
from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime


def get_image_crop_info(image_path):
    """
    crop the image in the center, ratio is 0.8
    Examples:
        input shape h*w is 304*320
        after crop the image, shape is 244*257
        left_up coordinate: (30, 31)
    """
    roi_vector = RoiBoxVector()
    roi = RoiBox()
    img = cv2.imread(image_path)
    central_fraction = 0.8
    height, width, _ = np.shape(img)
    target_h = int(height * central_fraction) + 1
    target_w = int(width * central_fraction) + 1
    amount_to_be_cropped_h = (height - target_h)
    amount_to_be_cropped_w = (width - target_w)
    crop_y = amount_to_be_cropped_h // 2
    crop_x = amount_to_be_cropped_w // 2
    print('crop image, width:{}, height:{}.'.format(width, height))
    print('crop image, x0:{}, y0:{}, x1:{}, y1:{}.'.format(crop_x, crop_y,
                                                           crop_x + target_w,
                                                           crop_y + target_h))
    roi.x0 = crop_x
    roi.y0 = crop_y
    roi.x1 = crop_x + target_w
    roi.y1 = crop_y + target_h
    roi_vector.push_back(roi)
    return roi_vector


def save_infer_result(result, result_name, image_name):
    """
    save the infer result to name_1.txt
    the file content top5:
        class_id1, class_id2, class_id3, class_id4, class_id5
    """
    load_dict = json.loads(result)
    if load_dict.get('MxpiClass') is None:
        with open(result_name + "/" + image_name[:-5] + '.txt', 'w') as f_write:
            f_write.write("")
    else:
        res_vec = load_dict['MxpiClass']
        with open(result_name + "/" + image_name[:-5] + '_1.txt', 'w') as f_write:
            list1 = [str(item.get("classId") - 1) + " " for item in res_vec]
            f_write.writelines(list1)
            f_write.write('\n')


if __name__ == '__main__':
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/inceptionv4_opencv.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = './val_union/'
    res_dir_name = 'result'
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(dir_name, file_name)
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()

        data_input.roiBoxs = get_image_crop_info(file_path)
        # Inputs data to a specified stream based on streamName.
        stream_name = b'im_inceptionv4'
        inplugin_id = 0
        unique_id = stream_manager.SendData(stream_name, inplugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager.GetResult(stream_name, unique_id)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # print the infer result
        print(infer_result.data.decode())
        save_infer_result(infer_result.data.decode(), res_dir_name, file_name)

    # destroy streams
    stream_manager.DestroyAllStreams()
