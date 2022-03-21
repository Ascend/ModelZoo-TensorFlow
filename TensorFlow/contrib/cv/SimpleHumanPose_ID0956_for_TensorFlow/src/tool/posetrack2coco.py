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


from npu_bridge.npu_init import *
import numpy as np
import json
import glob
import os
import os.path as osp
from PIL import Image
from pycocotools.coco import COCO

# run this code in the 'posetrack_data' folder

db = 'test' #train, val, test
annot_path = './annotations/' + db + '/'
filenames = glob.glob(annot_path + '*.json')
combined_annot = {'images': [], 'annotations': [], 'categories': []}
combined_annot_path = './combined_annotations/' + db + '2018.json'

for i in range(len(filenames)):

    with open(filenames[i]) as f:
        annot = json.load(f)
    
    for k,v in annot.items():
        if k == 'categories':
            combined_annot[k] = annot[k]

        elif k == 'images':
            for j in range(len(v)):
                imgname = v[j]['file_name']
                img = Image.open(osp.join('..', imgname))
                w,h = img.size
                annot[k][j]['width'] = w
                annot[k][j]['height'] = h
                annot[k][j]['coco_url'] = 'invalid'
            combined_annot[k] += annot[k]

        elif k == 'annotations':
            if db == 'train' or db == 'val':
                for j in range(len(v)):
                    annot[k][j]['num_keypoints'] = sum(annot[k][j]['keypoints'][2::3])
                    annot[k][j]['iscrowd'] = 0
                    if annot[k][j]['num_keypoints'] == 0:
                        annot[k][j]['bbox'] = [0,0,0,0]
                    annot[k][j]['area'] = annot[k][j]['bbox'][2] * annot[k][j]['bbox'][3]
            combined_annot[k] += annot[k]

        else:
            combined_annot[k] += annot[k]

        
with open(combined_annot_path, 'w') as f:
    json.dump(combined_annot, f)


