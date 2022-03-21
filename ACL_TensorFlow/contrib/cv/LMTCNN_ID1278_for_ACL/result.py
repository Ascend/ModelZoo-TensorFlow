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



import numpy as np
import os
import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('age_label_dir', './label/agelabel_1.npy', 'where to store the age testlabel')
tf.app.flags.DEFINE_string('gender_label_dir', './label/genderlabel_1.npy', 'where to store the gender testlabel')
tf.app.flags.DEFINE_string('output_dir', './output', 'where to store the testoutput')

if __name__ == '__main__': 
    agelabel = np.load(FLAGS.age_label_dir).astype(np.float32)
    genderlabel = np.load(FLAGS.gender_label_dir).astype(np.float32)
    agelabel = agelabel.reshape(len(agelabel))
    genderlabel = genderlabel.reshape(len(genderlabel))
    filelist = os.listdir(FLAGS.output_dir)
    filelist.sort()

    # print(len(filelist))
    age = np.zeros(1)
    gender = np.zeros(1)
    for i in range(0,len(filelist),2):
        # print(filelist[i],filelist[i+1])
        f1 = open(os.path.join(FLAGS.output_dir,filelist[i]))
        f2 = open(os.path.join(FLAGS.output_dir,filelist[i+1]))
        r1 = f1.readline()
        ages = r1.split()
        ages_float = map(float,ages)
        ages_float = np.array(list(ages_float))
        r2 = float(f2.readline())
        # print(r1,r2)
        if i ==0:
            age  = ages_float
            gender = r2
        else:
            age =np.vstack((age,ages_float))
            gender =np.vstack((gender,r2))
    
    gender = gender.reshape(len(gender))
    agetop1 = tf.nn.in_top_k(age, agelabel, 1)
    agetop2 = tf.nn.in_top_k(age, agelabel, 2)
    with tf.Session() as sess:
        agetop1,agetop2 = sess.run([agetop1,agetop2])
        agenum1 = np.sum(agetop1)
        agenum2 = np.sum(agetop2)

    print('age_top1: ',agenum1/len(agelabel),'age_top2: ',agenum2/len(agelabel))
    g = 0
    for i in range(0,len(agelabel)):
        if gender[i] == genderlabel[i]:
            g+=1
    print('gender: ',g/len(gender))
    