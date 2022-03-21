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

import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import pickle
import cv2
import os

eval_db_path = './datasets/faces_ms1m_112x112'
db_name = 'lfw'
image_size = [112, 112]

bins, issame_list = pickle.load(open(os.path.join(eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')

for i in range(len(issame_list)*2):
    _bin = bins[i]
    nparr = np.fromstring(_bin,np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    
    img = img - 127.5
    img = img * 0.0078125
    img = img.astype(np.float32)
    
    img.tofile('./bin_input/' + str(i) +'.bin')
    #img.tofile( str(i) +'.bin')
    
    #src = np.fromfile("0.bin", dtype=np.float32) 
    #src = np.reshape(src, [112,112,3])
    #cv2.imwrite('0.jpg', src)
    
    i += 1
    if i % 1000 == 0:
        print('converting to bin', i)


