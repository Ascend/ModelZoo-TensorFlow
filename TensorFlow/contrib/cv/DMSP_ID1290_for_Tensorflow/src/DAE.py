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

import time

from network import *
from PIL import Image
import tensorflow as tf
import scipy.misc as misc
import os


class DAE_MODEL:
    def __init__(self):
        self.clean_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.noised_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        dncnn = net("DnCNN")
        self.res = dncnn(self.noised_img, self.train_phase)
        self.denoised_img = self.noised_img - self.res
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.res - (self.noised_img - self.clean_img)), [1, 2, 3]))
        self.Opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,train_dir):
        filepath = train_dir
        filenames = os.listdir(filepath)
        print(filenames)
        saver = tf.train.Saver()
        for epoch in range(50):
            for i in range(filenames.__len__()//BATCH_SIZE):
                t = time.time()
                cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
                for idx, filename in enumerate(filenames[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]):
                    cleaned_batch[idx, :, :, :] = np.array(Image.open(filepath+'/'+filename).resize((IMG_H,IMG_W)))
                noised_batch = cleaned_batch + np.random.normal(0, SIGMA, cleaned_batch.shape)
                self.sess.run(self.Opt, feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: True})
                if i % 1 == 0:
                    [loss, denoised_img] = self.sess.run([self.loss, self.denoised_img], feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: False})
                    print("Epoch: %d, Step: %d, Loss: %g, Time %g"%(epoch, i, loss,time.time()-t))
                    compared = np.concatenate((cleaned_batch[0, :, :, 0], noised_batch[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    # Image.fromarray(np.uint8(compared)).save("./TrainingResults//"+str(epoch)+"_"+str(i)+".jpg")
                if i % 500 == 0:
                    saver.save(self.sess, "./save_para//DnCNN.ckpt")
            np.random.shuffle(filenames)

    def test(self, cleaned_path="./TestingSet//02.png"):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/DnCNN.ckpt")
        cleaned_img = np.reshape(np.array(misc.imresize(np.array(Image.open(cleaned_path)), [256, 256])), [1, 256, 256, 1])
        noised_img = cleaned_img + np.random.normal(0, SIGMA, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        Image.fromarray(np.uint8(compared)).show()



