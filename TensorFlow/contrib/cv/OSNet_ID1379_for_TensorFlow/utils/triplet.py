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
import math
import random
from sklearn.utils import shuffle as shuffle_tuple
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, Activation, Lambda, BatchNormalization, Input, concatenate, Embedding
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend as K
import cv2
from .general import seq

random.seed(2021)


def create_model(image_shape, num_person_ids, show_model_summary=False):
    anchor_input = Input(image_shape, name="anchor_input")
    positive_input = Input(image_shape, name="positive_input")
    negative_input = Input(image_shape, name="negative_input")

    cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling="max")
    cnn_model.trainable = False

    anchor_embedding = cnn_model(anchor_input)
    positive_embedding = cnn_model(positive_input)
    negative_embedding = cnn_model(negative_input)

    merged_vector = concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1, name="triplet")

    dense_anchor = Dense(num_person_ids)(anchor_embedding)
    softmax_anchor_output = Activation("softmax", name="softmax")(dense_anchor)
    
    triplet_model = Model([anchor_input, positive_input, negative_input], [merged_vector, softmax_anchor_output])

    if show_model_summary:
        triplet_model.summary()
    
    return triplet_model


def create_semi_hard_triplet_model(image_shape, num_person_ids, show_model_summary=False, resnet=True, last_stride_reduce=True, bn=True, center_loss=True, average_pooling=True):
    if resnet:
        print("Using model ResNet50V2\n")
        cnn_model = ResNet50V2(input_shape=image_shape, include_top=False, pooling=("avg" if average_pooling else "max"))
        if last_stride_reduce:
            cnn_model.get_layer("conv4_block6_2_conv").strides = (1,1)
            cnn_model.get_layer("max_pooling2d_2").strides = (1,1)
            cnn_model = model_from_json(cnn_model.to_json())
    else:
        print("Using model MobileNetV2\n")
        cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling=("avg" if average_pooling else "max"))
        if last_stride_reduce:
            cnn_model.get_layer("block_13_pad").padding = (1,1)
            cnn_model.get_layer("block_13_depthwise").strides = (1,1)
            cnn_model = model_from_json(cnn_model.to_json())

    global_pool = cnn_model.layers[-1].output
    cnn_model.layers[-1]._name = "triplet"
    
    if bn:
        features_bn = BatchNormalization(name="features_bn")(global_pool)
        dense = Dense(num_person_ids)(features_bn)
    else:
        dense = Dense(num_person_ids)(global_pool)
    softmax_output = Activation("softmax", name="softmax")(dense)
    
    if center_loss:
        input_target = Input(shape=(1,))
        centers = Embedding(num_person_ids, global_pool.shape[-1], name="embedding_center")(input_target)
        center_loss = Lambda(lambda x: 0.5 * K.sum(K.square((x[0] - x[1])), axis=1, keepdims=True), name="center")((global_pool, centers))
        triplet_model = Model([cnn_model.input, input_target], [global_pool, softmax_output, center_loss])
    else:
        triplet_model = Model(cnn_model.input, [global_pool, softmax_output])

    if show_model_summary:
        triplet_model.summary()
    
    return triplet_model
    
####
# Self defined triplet loss
####
def triplet_loss(y_true, y_pred, alpha=0.3):
    y_pred = K.l2_normalize(y_pred, axis=1)
    batch_num = y_pred.shape.as_list()[-1] // 3

    anchor = y_pred[:, 0:batch_num]
    positive = y_pred[:, batch_num:2*batch_num]
    negative = y_pred[:, 2*batch_num:3*batch_num]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)

    return loss

####
# Learning rate from paper
####
def lr_decay_warmup(epoch, initial_rate):
    epoch += 1
    if epoch < 11:
        return 3.5e-4 * epoch / 10
    elif epoch < 41:
        return 3.5e-4
    elif epoch < 71:
        return 3.5e-5
    else:
        return 3.5e-6
    


# Train data generator for cnn
class DataGenerator(Sequence):

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.
    def __init__(self, x_set, y_set, batch_size, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.shuffle:
            batch_x, batch_y = shuffle_tuple(batch_x, batch_y)

        if self.augment:
            batch_x = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in batch_x]).astype(np.uint8)
            batch_x = seq.augment_images(batch_x)
            batch_x = batch_x / 255.
        else:
            batch_x = np.array([np.asarray(load_img(file_path)) / 255. for file_path in batch_x])

        batch_x = (batch_x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch_y = to_categorical(np.array(batch_y), num_classes=self.num_classes)

        return batch_x, batch_y

# Train data generator for self defined triplet loss model
class DataGeneratorTriplet(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set

        # Make dict with key -> person_id, value -> list of associated images
        self.image_to_label = {}
        for image_path, image_label in zip(self.x, self.y):
            self.image_to_label.setdefault(image_label, []).append(image_path)

        # Get only anchor_id with more than 1 image
        self.anchor_filtered = [k for k, v in self.image_to_label.items() if len(v) > 1]

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        if self.shuffle:
            random.shuffle(self.anchor_filtered)
        
        # Get random sample of anchor_ids; amount: batch_size
        anchor_ids_sampled = random.sample(self.anchor_filtered, k=self.batch_size)
        # Get candidates of nagetive sample ids
        negative_id_cands = list(set(self.image_to_label.keys()) - set(anchor_ids_sampled))

        # Get anchor and positive image paths
        anchor_positive_list = [tuple(random.sample(self.image_to_label[id], k=2)) for id in anchor_ids_sampled]
        anchor_img_paths, positive_img_paths = zip(*anchor_positive_list)

        # Get negative image_paths
        negative_id_sampled = random.sample(negative_id_cands, k=self.batch_size)
        negative_img_paths = [random.choice(self.image_to_label[id]) for id in negative_id_sampled]

        if self.augment:
            anchor_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in anchor_img_paths]).astype(np.uint8)
            anchor_X_batch = seq.augment_images(anchor_X_batch)

            positive_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in positive_img_paths]).astype(np.uint8)
            positive_X_batch = seq.augment_images(positive_X_batch)

            negative_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in negative_img_paths]).astype(np.uint8)
            negative_X_batch = seq.augment_images(negative_X_batch)
            
        else:
            anchor_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in anchor_img_paths])
            positive_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in positive_img_paths])
            negative_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in negative_img_paths])

        anchor_X_batch = anchor_X_batch / 255.
        positive_X_batch = positive_X_batch / 255.
        negative_X_batch = negative_X_batch / 255.

        # Minus mean, devide by standard_deviation
        anchor_X_batch = (anchor_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        positive_X_batch = (positive_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        negative_X_batch = (negative_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        anchor_Y_batch = to_categorical(np.array(anchor_ids_sampled), num_classes=self.num_classes)

        return ([anchor_X_batch, positive_X_batch, negative_X_batch], [anchor_Y_batch, anchor_Y_batch])


# Train data generator for tensorflow-addons semihardtriplet loss model
class DataGeneratorHardTriplet(Sequence):
    def __init__(self, x_set, y_set, person_id_num, image_per_person_id, num_classes, shuffle=False, augment=False, center_loss=True):
        self.x, self.y = x_set, y_set

        # Make dict with key -> person_id, value -> list of associated images
        self.image_to_label = {}
        for image_path, image_label in zip(self.x, self.y):
            self.image_to_label.setdefault(image_label, []).append(image_path)

        # Get only anchor_id with at least `image_per_person_id`
        self.y_filtered = [k for k, v in self.image_to_label.items() if len(v) >= image_per_person_id]

        self.person_id_num = person_id_num
        self.image_per_person_id = image_per_person_id
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.center_loss = center_loss

    def __len__(self):
        return math.ceil(len(self.x) / (self.person_id_num * self.image_per_person_id))

    def __getitem__(self, idx):

        if self.shuffle: 
            random.shuffle(self.y_filtered)

        # Get random sample of ids; amount: `person_id_num`
        person_ids_chosen = random.sample(self.y_filtered, k=self.person_id_num)
        # For each id, get random sample of associate images; amount: `image_per_person_id`
        img_paths_sampled = [random.sample(self.image_to_label[id], k=self.image_per_person_id) for id in person_ids_chosen]
        img_paths_sampled = [path for paths in img_paths_sampled for path in paths]  # Flattening `img_paths_sampled`

        # Expand person_ids_chosen by `image_per_person_id` times to map with `img_paths_sampled`
        label_sampled = [[id] * self.image_per_person_id for id in person_ids_chosen]
        label_sampled = np.array([label for labels in label_sampled for label in labels])  # Flattening `label_sampled`

        if self.augment:
            X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in img_paths_sampled]).astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        else:
            X_batch = np.array([np.asarray(load_img(file_path)) for file_path in img_paths_sampled])

        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        Y_batch = to_categorical(np.array(label_sampled), num_classes=self.num_classes)

        if self.center_loss:
            return ([X_batch, label_sampled], [label_sampled, Y_batch, label_sampled])
        else:
            return (X_batch, [label_sampled, Y_batch])


# Test data generator
class DataGeneratorPredict(Sequence):

    def __init__(self, x_set, batch_size, shuffle=False, augment=False):
        self.x = x_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.augment:
            batch_x = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in batch_x]).astype(np.uint8)
            batch_x = seq.augment_images(batch_x)
            batch_x = batch_x / 255.
        else:
            batch_x = np.array([np.asarray(load_img(file_path)) / 255. for file_path in batch_x])

        batch_x = (batch_x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        return batch_x
        

def generator_batch_test(img_path_list, img_width, img_height, batch_size=32, shuffle=False):
    N = len(img_path_list)

    if shuffle:
        img_path_list = shuffle_tuple(img_path_list)

    batch_index = 0 # indicates batch_size

    while True:
        current_index = (batch_index*batch_size) % N #the first index for each batch per epoch
        if N >= (current_index+batch_size): # judge whether the current end index is over the train num
            current_batch_size = batch_size
            batch_index += 1 # indicates the next batch_size
        else:
            current_batch_size = N - current_index
            batch_index = 0
        img_batch_list = img_path_list[current_index:current_index + current_batch_size]

        X_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        for i, img_path in enumerate(img_batch_list):
            img = cv2.imread(img_path)
            if img.shape[:2] != (img_height, img_width):
                img = cv2.resize(img, (img_width, img_height))
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            X_batch[i, :, :, :] = img
        # normalization
        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        yield X_batch