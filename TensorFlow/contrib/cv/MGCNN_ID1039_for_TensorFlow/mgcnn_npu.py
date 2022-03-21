#
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
#
# Authors : Florian Lalande < florianlalande@orange.fr >
#           Austin Peel < austin.peel@cea.fr >

from npu_bridge.npu_init import *
import os
import sys
import time
import numpy as np
import tensorflow as tf
import argparse
from argparse import RawTextHelpFormatter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks

# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
from sklearn.metrics import confusion_matrix


def save_time(filepath, text=''):
    """Save current time to file."""
    hour = time.localtime().tm_hour
    min = time.localtime().tm_min
    sec = time.localtime().tm_sec
    with open(filepath, 'w') as f:
        f.write(text + "{:02}:{:02}:{:02}\n".format(hour, min, sec))


class History(callbacks.Callback):
    """Keras class used to track performance while training."""

    def on_train_begin(self, logs={}):
        self.losses = []
        self.valacc = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.valacc.append(logs.get('val_acc'))


class mgcnn(object):
    def __init__(self):
        self.model = self.create_model(verbose=True)

    def create_model(self, verbose=False):
        model = Sequential()
        model.add(Conv3D(8, (2, 3, 10), input_shape=(1, 4, 5, 100),
                         padding="same", activation='relu'))
        model.add(Conv3D(8, (2, 3, 10), padding="same", activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 1, 5)))
        model.add(Conv3D(8, (2, 3, 10), padding="same", activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 1, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        # Compile model
        optimizer = Adam(lr=0.001, decay=0.001)  # lr <- lr * (1.0/(1.0+decay*iter))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        if verbose:
            model.summary()
        return model

    def load_data(self, datapath, verbose=False):
        try:
            self.data = np.load(datapath)
            print("Loaded " + datapath)
        except Exception:
            print("Unable to load " + datapath)

    def reset(self, verbose=False):
        # Generate a new model for independent training
        del self.model
        self.model = self.create_model(verbose)
        if verbose:
            print("Resetting CNN model.")

    def train(self, savepath, iterations, epochs):
        if not hasattr(self, 'data'):
            print("No data loaded.")
            return

        save_time(os.path.join(savepath, "je_rentre.txt"), "start time ")

        X = self.data
        X = X.reshape(1024, 1, 4, 5, 100)

        y = np.floor(np.linspace(0, 3.9999, 1024))
        y = np_utils.to_categorical(y)

        h = np.zeros((iterations, 2, epochs))

        for iteration in range(iterations):
            print("\nITERATION {}".format(iteration))
            print("-" * (10 + len(str(iteration))))
            start_time = time.time()
            # Randomize training and test data
            i1 = np.arange(256)
            i2 = np.arange(256, 512)
            i3 = np.arange(512, 768)
            i4 = np.arange(768, 1024)

            np.random.shuffle(i1)
            np.random.shuffle(i2)
            np.random.shuffle(i3)
            np.random.shuffle(i4)

            i_train = np.concatenate((i1[64:], i2[64:], i3[64:], i4[64:]))
            i_test = np.concatenate((i1[:64], i2[:64], i3[:64], i4[:64]))
            np.random.shuffle(i_train)
            np.random.shuffle(i_test)

            X_train = X[i_train]
            y_train = y[i_train]
            X_test = X[i_test]
            y_test = y[i_test]

            # Do training and save performance measures
            history = History()
            self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                           epochs=epochs, batch_size=200, verbose=1,
                           callbacks=[history])
            h[iteration, 0] = history.losses
            h[iteration, 1] = history.valacc
            cb_path = os.path.join(savepath, "callback{}.npy".format(iteration))
            np.save(cb_path, h[iteration])

            # Compute confusion matrix
            y_pred = self.model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            cm = confusion_matrix(y_test, y_pred) / 64.0
            cm_path = os.path.join(savepath, "cm{}.npy".format(iteration))
            np.save(cm_path, cm)

            # Save model
            model_json = self.model.to_json()
            model_name = "model{}.json".format(iteration)
            with open(os.path.join(savepath, "model", model_name), "w") as f:
                f.write(model_json)
            weights = "weights{}.hdf5".format(iteration)
            self.model.save_weights(os.path.join(savepath, "model", weights))

            # Start next iteration fresh
            #self.reset()
            duration = time.time() - start_time
            print("time_per_iter %.3f " % duration)

        save_time(os.path.join(savepath, "je_sors.txt"), "end time ")


if __name__ == '__main__':
    # npu_keras_sess = set_keras_session_npu_config()
    # ***** npu modify begin *****
    os.system(" export ENABLE_FORCE_V2_CONTROL = 1")
    # export ENABLE_FORCE_V2_CONTROL = 1
    global_config = tf.ConfigProto(log_device_placement=False)
    custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_input"].b = 1
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    npu_keras_sess = set_keras_session_npu_config(config=global_config)
    # ***** npu modify end *****

    description = ("Train a CNN to classify four cosmological models\n" +
                   "based on their weak-lensing convergence maps.")
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("-n", "--noise", type=int, default=1,
                        help=("noise level; options are\n" +
                              "0 for clean data\n" +
                              "1 for sigma = 0.35\n" +
                              "2 for sigma = 0.70\n" +
                              "(default is 0)"))
    parser.add_argument("-i", "--iterations", type=int, default=100,
                        help="number of iterations (default is 100)")
    parser.add_argument("-e", "--epochs", type=int, default=1000,
                        help="number of epochs (default is 1000)")
    # parser.add_argument("-f", "--filename", type=str, default="all_pdfs.npy",
    parser.add_argument("-f", "--filename", type=str, default="pdf_j37.npy",
                        help="name of input data (default is all_pdfs.npy)")
    parser.add_argument("-d", "--subdir", type=str, default="subdir",
                        help="name of output sub-directory")
    parser.add_argument("-g", "--gpu", type=int, default=None,
                        help="specific gpu id to use")
    parser.add_argument("-datapath", "--datapath", type=str, default="/cache/dataset/data",
                        help="specific gpu id to use")
    args = parser.parse_args()


    noise_opts = {0: "clean", 1: "sigma035", 2: "sigma070"}
    noise_name = noise_opts[args.noise]

    # Only use a specific GPU
    ##if args.gpu is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        ##os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create output directories if needed
    cwd = os.getcwd()
    if not os.path.isdir("output"):
        os.mkdir("output")
    noise_dir = os.path.join("output", noise_name)#output/clean
    if not os.path.isdir(noise_dir):
        os.mkdir(noise_dir)
    sub_dir = os.path.join(noise_dir, args.subdir)#output/clean/subdir
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    model_dir = os.path.join(sub_dir, "model")#output/clean/subdir/model
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    outpath = os.path.join(cwd, sub_dir)#outpath=当前目录/output/clean/subdir/model

    if not os.path.isdir( outpath):
        os.mkdir( outpath)
    print("++++++")#标记

    # 在ModelArts容器创建训练输出目录
    model_dir = outpath  # "/cache/result"
    #os.makedirs(model_dir)

    # Build the CNN and train
    cnn = mgcnn()
    # datapath = os.path.join(cwd, "data", noise_name, args.filename)
    datapath = os.path.join(args.datapath, noise_name, args.filename)#/cache/dataset/data/clean/pdf_j15.npy,此处为数据路径
    #datapath = os.path.join(args.datapath, "clean/pdf_j15.npy")#/cache/dataset/data/clean/pdf_j15.npy,此处为数据路径
    cnn.load_data(datapath, verbose=True)
    print("Training on {} data for {} epochs and {} iterations.".format(
        noise_name, args.epochs, args.iterations))
    cnn.train(savepath=outpath, iterations=args.iterations, epochs=args.epochs)

    sys.exit()
    close_session(npu_keras_sess)

