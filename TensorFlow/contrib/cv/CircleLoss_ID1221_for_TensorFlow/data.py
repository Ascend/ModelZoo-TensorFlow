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
# from src.dataflow.market import MarketTriplet
# import skimage
# import skimage.transform
import numpy as np


def get_file_list(file_dir, file_ext, sub_name=None):
    """ Get file list in a directory with sepcific filename and extension

    Args:
        file_dir (str): directory of files
        file_ext (str): filename extension
        sub_name (str): Part of filename. Can be None.

    Return:
        List of filenames under ``file_dir`` as well as subdirectories

    """
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.lower().endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files)
            if name.lower().endswith(file_ext) and sub_name.lower() in name.lower()])



# def loadMarketForTest(data_dir='', rescale_im=[384, 128]):
#     def normalize_im(im):
#         im = skimage.transform.resize(
#             im, rescale_im,
#             mode='constant', preserve_range=True)
#         return np.clip(im/255.0, 0., 1.)

#     query_dir = os.path.join(data_dir, 'query')
#     query_data = MarketTripletForTest(
#         n_class=None,
#         data_dir=query_dir,
#         batch_dict_name=['im', 'label','camid'],
#         shuffle=True,
#         pf=normalize_im)
#     # query_data.setup(epoch_val=0, sample_n_class=10, sample_per_class=1)

#     test_dir = os.path.join(data_dir, 'bounding_box_test')
#     test_data = MarketTripletForTest(
#         n_class=None,
#         data_dir=test_dir,
#         batch_dict_name=['im', 'label','camid'],
#         shuffle=True,
#         pf=normalize_im)
#     # test_data.setup(epoch_val=0, sample_n_class=750, sample_per_class=2)
#     return query_data,test_data

# query_data,test_data = loadMarketForTest(market_dir, rescale_im=im_size)



# def loadMarket(P,K,data_dir='', rescale_im=[128, 64]):
#     def normalize_im(im):
#         im = skimage.transform.resize(
#             im, rescale_im,
#             mode='constant', preserve_range=True)
#         return np.clip(im/255.0, 0., 1.)
#
#     train_dir = os.path.join(data_dir, 'bounding_box_train')
#     train_data = MarketTriplet(
#         n_class=None,
#         data_dir=train_dir,
#         batch_dict_name=['im', 'label'],
#         shuffle=True,
#         pf=normalize_im)
#     train_data.setup(epoch_val=0, sample_n_class=P, sample_per_class=K)
#
#     test_dir = os.path.join(data_dir, 'bounding_box_test')
#     test_data = MarketTriplet(
#         n_class=None,
#         data_dir=test_dir,
#         batch_dict_name=['im', 'label'],
#         shuffle=True,
#         pf=normalize_im)
#     test_data.setup(epoch_val=0, sample_n_class=P, sample_per_class=K)
#
#     data = {}
#     data['test'] = test_data
#     data['train'] = train_data
#     return data

import numpy as np
import random
import tensorflow as tf

const_mean = np.zeros([256,128,3])
const_std = np.zeros([256,128,3])
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
for i in range(3):
    const_mean[:,:,i] = mean[i]
    const_std[:,:,i] = std[i]
const_mean = tf.constant(const_mean,dtype=tf.float32)
const_std = tf.constant(const_std,dtype=tf.float32)

# mean_tf1 = mean[0]*tf.ones(shape=[256,128,1])
# mean_tf2 = mean[1]*tf.ones(shape=[256,128,1])
# mean_tf3 = mean[2]*tf.ones(shape=[256,128,1])
# std_tf1 = std[0]*tf.ones(shape=[256,128,1])
# std_tf2 = std[1]*tf.ones(shape=[256,128,1])
# std_tf3 = std[2]*tf.ones(shape=[256,128,1])
# mean = tf.concat([mean_tf1,mean_tf2,mean_tf3],axis=2)
# std = tf.concat([std_tf1,std_tf2,std_tf3],axis=2)

def norm_trans(im):
    im = (im-const_mean)/const_std
    return im
def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    # HWC order
    import random
    if random.uniform(0, 1) > probability:
        return img
    # print('erasing')
    height = int(img.shape[0])
    width =  int(img.shape[1])
    channel =  int(img.shape[2])
    area = height*width

    erase_area_low_bound = round(np.sqrt(sl * area * r1))
    erase_area_up_bound = round((sh * area) / r1)
    h_upper_bound = min(erase_area_up_bound, height)
    w_upper_bound = min(erase_area_up_bound, width)

    h = np.random.randint(low=erase_area_low_bound,high=h_upper_bound)
    w = np.random.randint(low=erase_area_low_bound,high=w_upper_bound)

    x1 = np.random.randint(low=0,high=height+1-h)
    y1 = np.random.randint(low=0,high=width+1-w)

    mask = np.ones([height,width,channel])
    mask[x1:x1+h, y1:y1+w, :] = 0
    # print(f'erasing {h}x{w} pixel!')
    masktensor = tf.constant(mask,dtype=tf.float32)
    return img*masktensor




# from src.utils.dataflow import get_file_list
def tf_dataset_PK(P=4,K=4,erasing_prob=0.5,resize_image=[256,128],epoch_size=60,path='/home/nanshen/xutan/yixin/market1501/Market-1501-v15.09.15/bounding_box_train'):
    imlist = get_file_list(path, '.jpg')
    print(len(imlist))
    label = []
    for im_path in imlist:
        import ntpath
        head, tail = ntpath.split(im_path)
        class_label = tail.split('_')[0]
        label.append(int(class_label))
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(label)
    label_ = le.transform(label)
    # print(f'加载数据中一共有{len(le.classes_)}个类')
    # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string,channels=3)
        image_resized = tf.image.resize(image_decoded, resize_image,method=tf.image.ResizeMethod.BICUBIC)
        # print(image_resized.shape)
        # im = np.pad(image_resized,pad_width=((10,10),(10,10),(0,0)),mode='constant')
        image_padded = tf.image.pad_to_bounding_box(image_resized, 10,10,resize_image[0]+20,resize_image[1]+20)
        # im = random_crop(im,[256, 128])
        crop = tf.image.random_crop(image_padded,[resize_image[0],resize_image[1],3])
        rd = np.random.rand()
        if rd < 0.5:
            flip = tf.image.flip_left_right(crop)
        else:
            flip = crop
        im = tf.clip_by_value(flip/255.0,0,1.0)
        # norm
        im = norm_trans(im)
        # erasing
        im = random_erasing(im,probability=erasing_prob)
        return im, label

    imlists_dict = {}
    for i in range(751):
        imlists_dict[i] = []
    for indxi,labeli in enumerate(label_):
        imlists_dict[labeli].append(imlist[indxi])

    datasets = []
    for i in range(751):
        imlist_ind = imlists_dict[i]
        # 图片文件的列表
        filenames = tf.constant(imlist_ind)
        # label[i]就是图片filenames[i]的label
        label_ind = [i]*len(imlist_ind)
        labels = tf.constant(label_ind)

        # 此时dataset中的一个元素是(filename, label)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 此时dataset中的一个元素是(image_resized, label)
        dataset = dataset.map(_parse_function)

        # 此时dataset中的一个元素是(image_resized_batch, label_batch)
        dataset = dataset.shuffle(buffer_size=1000).batch(K).repeat(epoch_size)

        datasets.append(dataset)

    return [len(imlist),datasets,le]

# from src.utils.dataflow import get_file_list
def tf_dataset(erasing_prob=0.5,resize_image=[256,128],epoch_size=60,batch_size=32,path='/home/nanshen/xutan/yixin/market1501/Market-1501-v15.09.15/bounding_box_train'):
    imlist = get_file_list(path, '.jpg')
    print(len(imlist))
    label = []
    for im_path in imlist:
        import ntpath
        head, tail = ntpath.split(im_path)
        class_label = tail.split('_')[0]
        label.append(int(class_label))
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(label)
    label_ = le.transform(label)
    # print(f'加载数据中一共有{len(le.classes_)}个类')
    # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string,channels=3)
        image_resized = tf.image.resize(image_decoded, resize_image,method=tf.image.ResizeMethod.BICUBIC)
        # print(image_resized.shape)
        # im = np.pad(image_resized,pad_width=((10,10),(10,10),(0,0)),mode='constant')
        image_padded = tf.image.pad_to_bounding_box(image_resized, 10,10,resize_image[0]+20,resize_image[1]+20)
        # im = random_crop(im,[256, 128])
        crop = tf.image.random_crop(image_padded,[resize_image[0],resize_image[1],3])
        rd = np.random.rand()
        if rd < 0.5:
            flip = tf.image.flip_left_right(crop)
        else:
            flip = crop
        im = tf.clip_by_value(flip/255.0,0,1.0)
        # norm
        im = norm_trans(im)
        # erasing
        im = random_erasing(im,probability=erasing_prob)
        return im, label

    # 图片文件的列表
    filenames = tf.constant(imlist)
    # label[i]就是图片filenames[i]的label
    labels = tf.constant(label_)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(epoch_size)

    return [len(imlist),dataset,le]

# from src.utils.dataflow import get_file_list
def tf_val_dataset(resize_image=[256,128],epoch_size=60,batch_size=32,datapath='/home/liulizhao/projects/liuyixin/Market-1501-v15.09.15/'):
    queryimlist = get_file_list(datapath+'query', '.jpg')
    galleyimlist = get_file_list(datapath+'bounding_box_test', '.jpg')
    imlist = list(queryimlist) + list(galleyimlist)
    label = []
    for im_path in imlist:
        import ntpath
        head, tail = ntpath.split(im_path)
        class_label = tail.split('_')[0]
        label.append(int(class_label))
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(label)
    label_ = le.transform(label)
    # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string,channels=3)
        image_resized = tf.image.resize(image_decoded, resize_image,method=tf.image.ResizeMethod.BICUBIC)
        # print(image_resized.shape)
        # im = np.pad(image_resized,pad_width=((10,10),(10,10),(0,0)),mode='constant')
        # image_padded = tf.image.pad_to_bounding_box(image_resized, 10,10,resize_image[0]+20,resize_image[1]+20)
        # im = random_crop(im,[256, 128])
        # crop = tf.image.random_crop(image_padded,[resize_image[0],resize_image[1],3])
        # rd = np.random.rand()
        # if rd < 0.5:
        #     flip = tf.image.flip_left_right(crop)
        # else:
        #     flip = crop
        im = tf.clip_by_value(image_resized/255.0,0,1.0)
        # norm
        im = norm_trans(im)
        return im, label

    # 图片文件的列表
    filenames = tf.constant(imlist)
    # label[i]就是图片filenames[i]的label
    labels = tf.constant(label_)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(epoch_size)

    return dataset


import random
def random_crop(image, crop_shape):
    img_h = image.shape[0]
    img_w = image.shape[1]
    # img_d = image.shape[2]
    nh = random.randint(0, img_h - crop_shape[0])
    nw = random.randint(0, img_w - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1],:]
    return image_crop

# def loadMarketForTrain(P,K,data_dir='', rescale_im=[128, 64]):
#     def process_im(im):
#         im = skimage.transform.resize(
#             im, rescale_im,
#             mode='constant', preserve_range=True)
#         # pad
#         im = np.pad(im,pad_width=((10,10),(10,10),(0,0)),mode='constant')
#         # random crop
#         im = random_crop(im,rescale_im)
#         # horizontalFlip
#         im = np.fliplr(im)
#         # to tensor
#         im = np.clip(im/255.0, 0., 1.)
#         # normalize
#         mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         for i in range(3):
#             im[:,:,i] = (im[:,:,i]-mean[i])/std[i]
#         return im
#
#     train_dir = os.path.join(data_dir, 'bounding_box_train')
#     train_data = MarketTriplet(
#         n_class=None,
#         data_dir=train_dir,
#         batch_dict_name=['im', 'label'],
#         shuffle=True,
#         pf=process_im)
#     train_data.setup(epoch_val=0, sample_n_class=P, sample_per_class=K)
#
#     return train_data
    
def feed_dict(data,train,x,y_,lr,lr_new):
    """定义训练和测试操作"""
    if train:
        dict_ = data['train'].next_batch_dict()
        xs = dict_['im']
        ys = dict_['label']
    else:
        dict_ = data['test'].next_batch_dict()
        xs = dict_['im']
        ys = dict_['label']
    return {x: xs, y_: ys,lr:lr_new}

def get_next_batch(data):
    dict_ = data.next_batch_dict()
    xs = dict_['im']
    ys = dict_['label']
    return xs,ys

if __name__ == '__main__':
    valset = tf_val_dataset(batch_size=128,epoch_size=None)
    testiterator = valset.make_one_shot_iterator()
    testnext = testiterator.get_next()
        