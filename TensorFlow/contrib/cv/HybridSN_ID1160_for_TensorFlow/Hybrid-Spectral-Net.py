"""
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
"""


import tensorflow.keras
import tensorflow as tf
import npu_bridge
# from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv
import os

#os.system('pip install plotly')
#os.system('pip install spectral')
#from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#import spectral

# init_notebook_mode(connected=True)
# %matplotlib inline
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "train_url", "./result",
    "The output directory where the model checkpoints will be written.")

#flags.DEFINE_string("data_url", "./data/",
#                    "dataset path")
flags.DEFINE_string("datapath", "./data",
                     "dataset")

flags.DEFINE_integer(
    "batch_size", 32,
    "batch size for one NPU")

flags.DEFINE_integer(
    "train_step", 100,
    "total epochs for training")


## GLOBAL VARIABLES
dataset = 'SA'
test_ratio = 0.7
windowSize = 25


def loadData(name):               #根据变量dataset选择的数据集加载对应数据集，该加载数据集函数返回对应数据集及标签
    global data, labels
    # data_path = os.path.join(os.getcwd(), 'data')
    data_path = os.path.join(FLAGS.datapath, "data")
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    #划分训练集及测试集，返回训练集和测试集的数据集标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                            stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):              #实现数据集图片变换，并返回新数据集图片
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    #设置不同窗口大小，划分对应patch块，及patch块对应的标签大小，返回patch块的数据集和标签
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

with tf.compat.v1.Session(config=config) as sess:
    init_op = tf.group(tf.compat.v1.local_variables_initializer(), tf.compat.v1.global_variables_initializer())
    sess.run(init_op)

    X, y = loadData(dataset)

    print(X.shape, y.shape)

    K = X.shape[2]

    K = 30 if dataset == 'IP' else 15
    X, pca = applyPCA(X, numComponents=K)

    print(X.shape)

    X, y = createImageCubes(X, y, windowSize=windowSize)

    print(X.shape, y.shape)

    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

    print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    print(Xtrain.shape)

    # ytrain = np_utils.to_categorical(ytrain)
    ytrain = to_categorical(ytrain)
    print(ytrain.shape)

    S = windowSize
    L = K
    output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16

    ## input layer
    input_layer = Input((S, S, L, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    # print(conv_layer3._keras_shape)
    print(conv_layer3.shape)
    # conv3d_shape = conv_layer3._keras_shape
    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    # compiling the model
    adam = Adam(lr=0.001, decay=1e-06)
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    # # checkpoint
    # filepath = "best-model.hdf5"
    #
    # checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # # history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, callbacks=callbacks_list)
    # history = model.fit(x=Xtrain, y=ytrain, batch_size=FLAGS.batch_size, epochs=FLAGS.train_step, callbacks=callbacks_list)
    #
    #
    # # load best weights
    # model.load_weights("best-model.hdf5")
    # checkpoint
    # checkpoint_path = "training_1/cp.ckpt"
    checkpoint_path = FLAGS.train_url + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='acc', save_weights_only=True, verbose=1,
                                  mode='max')

    # Train the model with the new callback
    history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=FLAGS.train_step,
                        callbacks=[cp_callback])  # Pass callback to training

    # load best weights
    # model.load_weights("best-model.hdf5")
    model.load_weights(checkpoint_path)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    print(Xtest.shape)

    # ytest = np_utils.to_categorical(ytest)
    ytest = to_categorical(ytest)
    print(ytest.shape)

    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)

    classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
    print(classification)


    def AA_andEachClassAccuracy(confusion_matrix):
        #根据混淆矩阵计算准确率及其平均准确率，返回准确率和平均准确率
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc


    def reports(X_test, y_test, name):
        #根据dataset所选取的数据集加载对应数据集的标签，并预测测试集准确率，oa，混淆矩阵，平均准确率，Kappa，训练损失，训练准确率
        # start = time.time()
        Y_pred = model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        # end = time.time()
        # print(end - start)
        if name == 'IP':
            target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                            'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                            'Stone-Steel-Towers']
        elif name == 'SA':
            target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                            'Fallow_smooth',
                            'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                            'Corn_senesced_green_weeds',
                            'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                            'Vinyard_untrained', 'Vinyard_vertical_trellis']
        elif name == 'PU':
            target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                            'Self-Blocking Bricks', 'Shadows']

        classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
        oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
        confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
        score = model.evaluate(X_test, y_test, batch_size=32)
        Test_Loss = score[0] * 100
        Test_accuracy = score[1] * 100
        # namedtuple = classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc *100, aa *100, kappa *100
        # return namedtuple

        return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100


    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest, ytest, dataset)
    classification = str(classification)
    confusion = str(confusion)
    file_name = FLAGS.train_url + "classification_report.txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


    def Patch(data, height_index, width_index):         #设置生成预测图窗口的大小
        height_slice = slice(height_index, height_index + PATCH_SIZE)
        width_slice = slice(width_index, width_index + PATCH_SIZE)
        patch = data[height_slice, width_slice, :]

        return patch


    # load the original image
    X, y = loadData(dataset)

    height = y.shape[0]
    width = y.shape[1]
    PATCH_SIZE = windowSize
    numComponents = K

    X, pca = applyPCA(X, numComponents=numComponents)

    X = padWithZeros(X, PATCH_SIZE // 2)

    # calculate the predicted image
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j)
                X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                   1).astype('float32')
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction + 1

    #ground_truth = spectral.imshow(classes=y.astype(int), figsize=(7, 7))

    #predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))

    #spectral.save_rgb(FLAGS.train_url + "predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)
    #spectral.save_rgb(FLAGS.train_url + str(dataset) + "_ground_truth.jpg", y, colors=spectral.spy_colors)
