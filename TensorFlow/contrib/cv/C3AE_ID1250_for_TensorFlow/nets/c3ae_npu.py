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
# Copyright 2022 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, ReLU, \
AveragePooling2D, Concatenate, Dense, Flatten, Lambda, GlobalAveragePooling2D, multiply
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.activations import softmax,sigmoid
from tensorflow.python.keras.optimizers import Adam
from utils import focal_loss
import os



def config_gpu():
    from tensorflow.python.keras.backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def BRA(input):
    bn = BatchNormalization()(input)
    activation = ReLU()(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)

def BN_ReLU(input, name):
    return ReLU()(BatchNormalization()(input))

def preprocessing(dataframes, batch_size=50, category=12, interval=10, is_training=True, dropout=0.):
    # category: bin + 2 due to two side
    # interval: age interval
    from utils import generate_data_generator
    return generate_data_generator(dataframes, category=category, interval=interval, batch_size=batch_size,
                                   is_training=is_training, dropout=dropout)

def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation=sigmoid)(ReLU()(fc1))
    return multiply([scale, input])

def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def build_shared_plain_network(height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    input_image = Input(shape=(height, width, channel))

    if using_white_norm:
        wn = Lambda(white_norm, name="white_norm")(input_image)
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(wn)
    else:
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(input_image)
    block1 = BRA(conv1)
    block1 = SE_BLOCK(block1, using_SE)

    conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2")(block1)
    block2 = BRA(conv2)
    block2 = SE_BLOCK(block2, using_SE)

    conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3")(block2)
    block3 = BRA(conv3)
    block3 = SE_BLOCK(block3, using_SE)

    conv4 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv4")(block3)
    block4 = BN_ReLU(conv4, name="BN_ReLu")
    block4 = SE_BLOCK(block4, using_SE)

    conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)
    conv5 = SE_BLOCK(conv5, using_SE)

    flat_conv = Flatten()(conv5)

    pmodel = Model(inputs=input_image, outputs=[flat_conv])
    return pmodel

def build_net(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    print(base_model.summary())
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    bulk_feat = Dense(CATES, use_bias=True, activity_regularizer=regularizers.l1(0), activation=softmax, name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)

    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat])

def train(params):
    from utils import reload_data
    sample_rate, seed, batch_size, category, interval = 0.7, 2019, params.batch_size, params.category + 2, int(
        math.ceil(100. / params.category))

    lr = params.learning_rate
    data_dir, file_ptn = params.dataset, params.source
    dataframes = reload_data(data_dir, file_ptn)
    # print(dataframes)

    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1 - sample_rate,
                                         random_state=seed)
    # print("train_shape:", trainset.shape)
    # print(trainset.groupby(["age"])["age"].agg("count"))

    train_gen = preprocessing(trainset, dropout=params.dropout, category=category, interval=interval)
    validation_gen = preprocessing(testset, is_training=False, category=category, interval=interval)

    # print(testset.groupby(["age"]).agg(["count"]))
    age_dist = [trainset["age"][(trainset.age >= x - 10) & (trainset.age <= x)].count() for x in range(10, 101, 10)]
    age_dist = [age_dist[0]] + age_dist + [age_dist[-1]]
    # print(age_dist)

    if params.pretrain_path and os.path.exists(params.pretrain_path):
        models = build_net(category, using_SE=params.se_net, using_white_norm=params.white_norm)
        models.load_weights(params.pretrain_path)
    else:
        models = build_net(category, using_SE=params.se_net, using_white_norm=params.white_norm)
    adam = Adam(lr=lr)

    models.compile(
        optimizer=adam,
        loss=["mae", focal_loss(age_dist)],
        metrics={"age": "mae"},
        loss_weights=[1, params.weight_factor]
    )

    print(models.summary())
    save_path=os.path.join(params.save_path,'c3ae_npu_train.h5')

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_age_mean_absolute_error', verbose=2, save_best_only=True,
                        save_weights_only=False, mode='min'),
    ]

    history = models.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=params.epochs,
                                   callbacks=callbacks, validation_data=validation_gen,
                                   validation_steps=len(testset) / batch_size)


def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='C3AE retry')
    parser.add_argument(
        '-s', '--save_path', default="./model/c3ae_npu_train.h5", type=str,
        help='the best model to save')
    parser.add_argument(
        '-r', '--r_factor', default=2, type=int,
        help='the r factor of SE')
    parser.add_argument(
        '--source', default="wiki", type=str,
        choices=['wiki', 'imdb', 'wiki|imdb'],
        help='"wiki|imdb" or regrex pattern of feather')

    parser.add_argument(
        '--dataset', default="./dataset/data/", type=str,
        help='the path of dataset to load')

    parser.add_argument(
        '-p', '--pretrain_path', dest="pretrain_path", default="", type=str,
        help='the pretrain path')

    parser.add_argument(
        '-b', '--batch_size', default=256, type=int,
        help='batch size degfault=256')

    parser.add_argument(
        '-w', '--weight_factor', default=10, type=int,
        help='age feature weight=10')


    parser.add_argument(
        '-c', '--category', default=10, type=int,
        help='category nums degfault=10, n+2')

    parser.add_argument(
        '-d', '--dropout', default="0.2", type=float,
        help='dropout rate of erasing')

    parser.add_argument(
        '-lr', '--learning-rate', default="0.002", type=float,
        help='learning rate')

    parser.add_argument(
        '-se', "--se-net", dest="se_net", action='store_true',
        help='use SE-NET')

    parser.add_argument(
        '-white', '--white-norm', dest="white_norm", action='store_true',
        help='use white norm')

    parser.add_argument(
        '-gpu', dest="gpu", action='store_true',
        help='config of GPU')
    parser.add_argument(
        '-epochs', '--epochs', default=600, type=int,
        help='train epochs')


    params = parser.parse_args()

    if params.gpu:
        config_gpu()
    return params


if __name__ == "__main__":
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add() 
    custom_op.name = "NpuOptimizer" 
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF 
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    params = init_parse()
    train(params)
    sess.close()
