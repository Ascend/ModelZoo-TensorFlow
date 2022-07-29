#encoding=utf-8
#
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
#
from npu_bridge.npu_init import *
import numpy as np;
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
import config
        

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print("%d images found in %s"%(len(image_names), data_path))
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            path = util.io.join_path(data_path, image_name);
            print("\tconverting image: %d/%d %s"%(idx, len(image_names), image_name))
            image_data = tf.gfile.FastGFile(path, 'rb').read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];
            gt_name = 'gt_' + image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
                
            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                #line = util.str.remove_all(line, '\ufeff')
                #line = line.strip('\n')
                gt = util.str.split(line, ',');
                oriented_box = [int(gt[i]) for i in range(8)];
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4);
                oriented_bboxes.append(oriented_box);
                
                xs = oriented_box.reshape(4, 2)[:, 0]                
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1]); 
                ignored = util.str.contains(gt[-1], '###')
                if ignored:
                    labels.append(config.ignore_label);
                else:
                    labels.append(config.text_label)
            example = convert_to_example(image_data, image_name, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('./datasets/icdar2015/')
    # root_dir = util.io.get_absolute_path(FLAGS.root_dir)
    output_dir = util.io.get_absolute_path('./datasets/ICDAR2015/')
    # output_dir = util.io.get_absolute_path(FLAGS.output_dir)
    util.io.mkdir(output_dir)

    training_data_dir = util.io.join_path(root_dir, 'ch4_training_images')
    training_gt_dir = util.io.join_path(root_dir,'ch4_training_localization_transcription_gt')
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2015_train.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)

    test_data_dir = util.io.join_path(root_dir, 'ch4_test_images')
    test_gt_dir = util.io.join_path(root_dir,'Challenge4_Test_Task1_GT')
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2015_test.tfrecord'), data_path = test_data_dir, gt_path = test_gt_dir)
