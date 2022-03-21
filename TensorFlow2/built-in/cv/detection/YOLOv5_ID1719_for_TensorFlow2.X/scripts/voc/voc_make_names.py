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

import sys
import os

from absl import app, flags
from absl.flags import FLAGS
from lxml import etree


flags.DEFINE_string('anno_dir', '../../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations', 'path to anno dir')
flags.DEFINE_string('output', '../../data/classes/voc2012.names', 'path to anno dir')


def make_names(anno_dir, output):
    labels_dict = {}

    anno_list = os.listdir(anno_dir)

    for anno_file in anno_list:
        p = os.path.join(anno_dir, anno_file)
        
        # Get annotation.
        root = etree.parse(p).getroot()
        names = root.xpath('//object/name')

        for n in names:
            labels_dict[n.text] = 0
    
    labels = list(labels_dict.keys())
    labels.sort()

    with open(output, 'w') as f:
        for l in labels:
            f.writelines(l + '\n')

    print(f"Done making a names's file ({os.path.abspath(output)})")


def main(_argv):
    make_names(FLAGS.anno_dir, FLAGS.output)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass    
