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


import os
import numpy as np
import glob
import argparse
import lpips_tf
import tensorflow as tf
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str,default="/root/output",help="")

args = parser.parse_args()

result=[]
for root,dirs,files in os.walk(args.output_path):
    for index,dir in enumerate(dirs):
        print(dir)
        result.append(sorted(glob.glob(os.path.join(root,dir,'*.bin'))))

file_nums = len(result[0])

result = np.array(result)
result = np.transpose(result,[1,0])

with tf.Session() as sess:
    dist_consecutive=[]
    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)
    dist_t = lpips_tf.lpips(image0_ph,image1_ph)

    for arr in tqdm.tqdm(result):
        lpips_pair =[]
        for path in arr:
            image = np.fromfile(path,dtype=np.float32)
            image = (image + 1)/2
            image = np.reshape(image,(256,256,3))
            lpips_pair.append(image)
        for i in range(0,len(lpips_pair)-1):
            dist= sess.run(dist_t,feed_dict={image0_ph:lpips_pair[i],image1_ph:lpips_pair[i+1]})
            dist_consecutive.append(dist)
    
    print('Final Average Distances : {}'.format(sum(dist_consecutive)/len(dist_consecutive)))











