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

"""Create image-list file
Example:
python tools/create_img_lists.py --dataset=data/celebA --outfile=data/train.txt
"""
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dataset", dest="dataset",  
                  help="dataset path")

parser.add_option("--outfile", dest="outfile",  
                  help="outfile path")
(options, args) = parser.parse_args()

# options.dataset = '/root/data/celebA'
# options.outfile = '/root/data/train.txt'
options.dataset = './celebA'                 #sed修改
print("--------------------",options.dataset)
options.outfile = './train.txt'

f = open(options.outfile, 'w')
dataset_basepath = options.dataset
for p1 in os.listdir(dataset_basepath):
  image = os.path.abspath(dataset_basepath + '/' + p1)
  f.write(image + '\n')
f.close()
