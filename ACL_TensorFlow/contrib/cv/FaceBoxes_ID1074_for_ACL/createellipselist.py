#-*-coding:utf-8-*-
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
import shutil
import sys
sys.path.append('.')
import os
import argparse





#TODO change parameters
ap = argparse.ArgumentParser()
ap.add_argument( "--split_dir", required=False,default='../FDDB/FDDB-folds/FDDB-folds/',help="dir to FDDB-folds")
ap.add_argument( "--result", required=False,default='../FDDBFile',help="dir to write result")
args = ap.parse_args()


ANNOTATIONS_PATH = args.split_dir
RESULT_DIR = args.result

if not os.access(RESULT_DIR,os.F_OK):
    os.mkdir(RESULT_DIR)

annotations = [s for s in os.listdir(ANNOTATIONS_PATH) if s.endswith('ellipseList.txt')]
image_lists = [s for s in os.listdir(ANNOTATIONS_PATH) if not s.endswith('ellipseList.txt')]
annotations = sorted(annotations)
image_lists = sorted(image_lists)

images_to_use = []
for n in image_lists:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        images_to_use.extend(f.readlines())

images_to_use = [s.strip() for s in images_to_use]
with open(os.path.join(RESULT_DIR, 'faceList.txt'), 'w') as f:
    for p in images_to_use:
        f.write(p + '\n')


ellipses = []
for n in annotations:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        ellipses.extend(f.readlines())

i = 0
with open(os.path.join(RESULT_DIR, 'ellipseList.txt'), 'w') as f:
    for p in ellipses:

        # check image order
        if 'big/img' in p:
            assert images_to_use[i] in p
            i += 1

        f.write(p)



