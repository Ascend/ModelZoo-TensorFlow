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
#
# ==============================================================================
import os
import argparse
import xml.etree.ElementTree as ET


def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
               'bus', 'car', 'cat', 'chair', 'cow', 
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for idx, image_ind in enumerate(image_inds):
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            
            print('idx=', idx, ' annotation=', annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/avBuffer/VoAI/VOC/")
    parser.add_argument("--train_annotation", default="./imgs/dataset/voc_train.txt")
    parser.add_argument("--test_annotation",  default="./imgs/dataset/voc_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):
        os.remove(flags.train_annotation)
    
    if os.path.exists(flags.test_annotation):
        os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2007'), 'trainval', flags.train_annotation, False)
    num2 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2012'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(os.path.join(flags.data_path, 'test/VOCdevkit/VOC2007'),  'test', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (num1 + num2, num3))
