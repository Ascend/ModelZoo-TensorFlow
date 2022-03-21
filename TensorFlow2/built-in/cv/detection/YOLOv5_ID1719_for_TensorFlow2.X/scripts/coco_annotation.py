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

from absl import app, flags, logging
import os
import pickle
from os import listdir
from os.path import isfile, join
from absl.flags import FLAGS
import cv2

flags.DEFINE_string('coco_data', './val2017.pkl', 'path to coco data')
flags.DEFINE_string('classes', '../data/classes/coco.names', 'path to classes file')
flags.DEFINE_string('coco_path', "/Volumes/Elements/data/coco_dataset/coco", 'resize images to')
flags.DEFINE_string('image_path', "images/val2017", 'path to image val')
flags.DEFINE_string('anno_path_val', '../data/dataset/val2017.txt', 'path to classes file')

def convert_annotation(output, data, data_type = "val"):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    replace_dict = {"couch": "sofa", "airplane": "aeroplane", "tv": "tvmonitor", "motorcycle": "motorbike"}

    if os.path.exists(output): os.remove(output)
    directory_path = os.path.join(FLAGS.coco_path, FLAGS.image_path)
    # if data_type == "train":
    #     anno_path = directory_path + "/labels/train2014"
    #     image_path = os.path.join(directory_path, "trainvalno5k.txt")
    # else:
    #     anno_path = directory_path + "/labels/val2014"
    #     image_path = os.path.join(directory_path, "5k.txt")
    # with open(image_path) as f:
    #     image_paths = f.readlines()
    # image_paths = [x.strip() for x in image_paths]

    image_paths = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

    check_classes = []
    count = 0
    with open(output, 'a') as f:
        for image_path in image_paths:
            image_inds = image_path.split(".")[0]
            annotation = os.path.join(directory_path, image_path)
            # if os.path.exists(os.path.join(anno_path, image_inds + ".txt")):
            if image_inds in data:
                objects = data[image_inds]["objects"]
                for key, value in objects.items():
                    if key == 'num_obj': continue
                    if value["name"] not in class_names:
                        class_ind = replace_dict[value["name"]]
                        class_ind = class_names.index(class_ind)
                        # if value["name"] not in check_classes:
                        #     check_classes.append(value["name"])
                        #     print(value["name"])
                        # continue
                    else:
                        class_ind = class_names.index(value["name"])
                    xmin = int(value["bndbox"]["xmin"])
                    xmax = int(value["bndbox"]["xmax"])
                    ymin = int(value["bndbox"]["ymin"])
                    ymax = int(value["bndbox"]["ymax"])
                    annotation += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_ind)])
            else: continue
            f.write(annotation + "\n")
            count += 1
            # print(annotation)
    print(count)
    return

def main(_argv):
    with open(FLAGS.coco_data, "rb") as input_file:
        data = pickle.load(input_file)
    data = data[1]
    convert_annotation(FLAGS.anno_path_val, data)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass