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
import tensorflow as tf
from os.path import join

class LmserData_train():
    def __init__(self,npy_path, train_dir):
        self.npy_path = npy_path
        self.train_dir = train_dir


    def get_data(self):
        D_data = np.load(join(self.train_dir, 'D_wider_lnew.npy'), allow_pickle=True)
        G_data = np.load(self.npy_path, allow_pickle=True).item()

        np.random.shuffle(D_data)
        input_img, lr_img, hr_img = G_data['input_img'], G_data['lr_img'], G_data['hr_img']
        merge = list(zip(input_img, lr_img, hr_img))
        np.random.shuffle(merge)
        input_img, lr_img, hr_img = zip(*merge)
        input_img = np.array(input_img)
        lr_img = np.array(lr_img)
        hr_img = np.array(hr_img)
        return D_data, input_img, lr_img, hr_img

    def flip_left_right(self, img):
        return img[:,::-1,:]

    def get_next(self, batch_id, batch_size, D_data, input_img, lr_img, hr_img):
        last_index = batch_id + batch_size
        g_d_data = D_data[batch_id: last_index]
        g_input_img = input_img[batch_id: last_index]
        g_lr_img = lr_img[batch_id: last_index]
        g_hr_img = hr_img[batch_id: last_index]
        for i in range(len(g_d_data)):
            if np.random.random() > 0.5:
                g_d_data[i] = self.flip_left_right(g_d_data[i])
                g_input_img[i] = self.flip_left_right(g_input_img[i])
                g_lr_img[i] = self.flip_left_right(g_lr_img[i])
                g_hr_img[i] = self.flip_left_right(g_hr_img[i])
        return g_d_data, g_input_img, g_lr_img, g_hr_img, last_index

class LmserData_test():
    def __init__(self, test_dir):
        self.test_dir = test_dir

    def get_data(self):
        data = np.load(join(self.test_dir, '0.npy'), allow_pickle=True).item()
        input_img, lr_img, hr_img = data['input_img'], data['lr_img'], data['hr_img']
        return input_img, lr_img, hr_img

    def get_next(self, batch_id, batch_size, input_img, lr_img, hr_img):
        last_index = batch_id + batch_size
        t_input_img = input_img[batch_id:last_index]
        t_lr_img = lr_img[batch_id:last_index]
        t_hr_img = hr_img[batch_id:last_index]
        return t_input_img, t_lr_img, t_hr_img, last_index

    def get_d_data(self, d_data, batch_id, batch_size):
        last_index = batch_id + batch_size
        return d_data[batch_id:last_index]
        


