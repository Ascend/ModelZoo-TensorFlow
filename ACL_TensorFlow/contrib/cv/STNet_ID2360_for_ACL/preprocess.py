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
import numpy as np
from PIL import Image
import os
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--total_num', default=200,
                    help="""number of total images""")
parser.add_argument('--data_path', default='./dataset/data.npz',
                    help="""dataset path""")          
args, unknown_args = parser.parse_known_args()          
mnist_cluttered = np.load(args.data_path)
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']
tot=int(args.total_num)
if not os.path.isdir('./images'):
    os.makedirs(os.path.realpath('./images'))
for i in range(tot):
    im=X_test[i]
    im=im.reshape(40,40)
    im *= 255 
    im = Image.fromarray(im)
    im = im.convert('L')  
    im.save('./images/'+str(i)+'.png')
