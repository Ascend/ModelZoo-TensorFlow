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
import numpy as np
import h5py
import os
import cv2
import random
import sys
import tensorflow as tf




def prepare_data(path):
    f = h5py.File('%s/cuhk-03.mat' % path)
    labeled = [f['labeled'][0][i] for i in range(len(f['labeled'][0]))]
    labeled = [f[labeled[0]][i] for i in range(len(f[labeled[0]]))]
    detected = [f['detected'][0][i] for i in range(len(f['detected'][0]))]
    detected = [f[detected[0]][i] for i in range(len(f[detected[0]]))]
    datasets = [['labeled', labeled], ['detected', detected]]
    prev_id = 0

    for dataset in datasets:
        if not os.path.exists('%s/%s/train' % (path, dataset[0])):
            os.makedirs('%s/%s/train' % (path, dataset[0]))
        if not os.path.exists('%s/%s/val' % (path, dataset[0])):
            os.makedirs('%s/%s/val' % (path, dataset[0]))

        for i in range(0, len(dataset[1])):
            for j in range(len(dataset[1][0])):
                try:
                    image = np.array(f[dataset[1][i][j]]).transpose((2, 1, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = cv2.imencode('.jpg', image)[1].tostring()
                    if len(dataset[1][0]) - j <= 100:
                        filepath = '%s/%s/val/%04d_%02d.jpg' % (path, dataset[0], j - prev_id - 1, i)
                    else:
                        filepath = '%s/%s/train/%04d_%02d.jpg' % (path, dataset[0], j, i)
                        prev_id = j
                    with open(filepath, 'wb') as image_file:
                        image_file.write(image)
                except Exception as e:
                    continue

def get_pair(path, set, num_id,select_id_one_value,images_per_id):

    pair = []
    #positive and anchor
    #value = int(random.random() * num_id)
    #id = [select_id, select_id]

    index_overlap =[]
      
    label_anchor = []
    label_anchor = np.zeros(num_id)
    label_anchor[select_id_one_value-1] = 1 # for softmax classification
    
    for i in range(images_per_id):
        filepath = ''
        
        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, select_id_one_value, index)
            if not os.path.exists(filepath):
                continue
            if index in index_overlap:
                continue
            break
        index_overlap.append(index)
        pair.append(filepath)

    
    return pair,label_anchor


def get_num_id(path, set):
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1


def read_data(path, set, num_id, image_width, image_height, batch_size,select_id_num,images_per_id):
    batch_images = []
    labels = []
    labels_neg = []
    labels_two = [] # it is [0 1]
    labels_name = []
    
    #select_id_num = 6  # num_id is total training data id
    #images_per_id   # img  per  idZZ
                
    select_id = random.sample(range(num_id), select_id_num)
    
    batch_images_total = []
    
    for i in range(select_id_num):
        #pairs = [get_pair(path, set, num_id, True), get_pair(path, set, num_id, False)]
        
        pairs , label_anchor = get_pair(path, set, num_id,select_id[i],images_per_id)
        

        #print(pairs)
        #print(num_id)#'2'
        #print(pairs)#[['data2/labeled/train/0000_00.jpg', 'data2/labeled/train/0000_01.jpg'], ['data2/labeled/train/0001_00.jpg', 'data2/labeled/train/0000_01.jpg']]
        images = []
        for pair in pairs:
            image = cv2.imread(pair)
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            #print pair
            batch_images_total.append(image)
        batch_images.append(images)
            
        labels.append(label_anchor)
        labels_two.append([0., 1.])
    
        for la in range(images_per_id):
            labels_name.append(select_id[i])
    
    
    
    
    
    '''
    for pair in batch_images:
        for p in pair:
            cv2.imshow('img', p)
            key = cv2.waitKey(0)
            if key == 1048603:
                exit()
    '''
    

    #print(np.transpose(batch_images, (1, 0, 2, 3, 4)).shape)
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels_name),np.array(batch_images_total)

    #return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels),np.array(labels_neg),np.array(labels_two)










'''

def get_pair2(path, set, num_id, positive):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break

    for i in xrange(2):
        filepath = ''
        while True:
            index = int(random.random() * 10)
            filepath = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id[i], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    print(pair)
    return pair


def read_data2(path, set, num_id, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in xrange(batch_size // 2):
        pairs = [get_pair2(path, set, num_id, True), get_pair2(path, set, num_id, False)]
        print(np.array(pairs).shape)
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            print(np.array(images).shape)
            batch_images.append(images)
        labels.append([1., 0.])
        labels.append([0., 1.])


    print(np.array(batch_images).shape)
    print(np.transpose(batch_images, (1, 0, 2, 3, 4)).shape)
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)
    
'''
    
    
    
    
    
    
    
    


    
    

if __name__ == '__main__':
    #prepare_data(sys.argv[1])
    
    
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer('batch_size', '10', 'batch size for training')
    tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
    tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
    tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
    tf.flags.DEFINE_float('learning_rate', '0.01', '')
    tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
    tf.flags.DEFINE_string('image1', '', 'First image path to compare')
    tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
    
    tf.flags.DEFINE_integer('ID_num', '5', 'id number')
    tf.flags.DEFINE_integer('IMG_PER_ID', '2', 'img per id')

    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 160
    batch_images, batch_labels,batch_images_total = read_data(FLAGS.data_dir, 'train', 700,IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    print (batch_images.shape)
    print (batch_labels.shape)
    print (batch_images_total.shape)
    print (batch_labels)


    '''
    ['data//labeled/train/0658_03.jpg', 'data//labeled/train/0658_05.jpg', 'data//labeled/train/0641_07.jpg']
    ['data//labeled/train/0283_01.jpg', 'data//labeled/train/0283_08.jpg', 'data//labeled/train/0565_06.jpg']
    (3, 2, 224, 224, 3)
    '''             