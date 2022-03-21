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

''' scanning through annotation files for all the scenes to get a complete list of categories '''

import os
import json
scannet_dir = './scannet/'
scene_names = [line.rstrip() for line in open('scannet_all.txt')]

labels = set()
for scene_name in scene_names:
    path = os.path.join(scannet_dir, scene_name)
    agg_filename = os.path.join(path, scene_name+'.aggregation.json')
    with open(agg_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            labels.add(x['label']) 

fout = open('class_names.txt', 'w')
for label in list(labels):
    print label
    try:
        fout.write(label+'\n')
    except:
        pass
fout.close()
