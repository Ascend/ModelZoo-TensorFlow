
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
#import keras
#from keras.models import Model
#from keras.layers import Dense, Dropout
#from keras.callbacks import ModelCheckpoint, TensorBoard
#from keras.optimizers import SGD, Adam
#from util.data_loader import train_generator, val_generator
#import keras.backend.tensorflow_backend as K
#from keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam


from util.data_loader import train_generator, val_generator

#from npu_bridge.estimator.npu import npu_convert_dropout

# import moxing as mox

import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from absl import flags
FLAGS =tf.app.flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


image_size = 224

tf.app.flags.DEFINE_integer('batchsize',100,'training batch')
tf.app.flags.DEFINE_integer('epochs',7,'training epoch')
tf.app.flags.DEFINE_integer('train_size',250502,'training size')
tf.app.flags.DEFINE_integer('val_size',5000,'testing size')
tf.app.flags.DEFINE_integer('image_size',224,'testing size')
tf.app.flags.DEFINE_string('precision_mode','allow_mix_precision','testing size')
tf.app.flags.DEFINE_string('TMP_WEIGHTS_PATH','./weights','testing size')
tf.app.flags.DEFINE_string('TMP_MODEL_PATF','./model','testing size')
tf.app.flags.DEFINE_string('TMP_LOG_PATH','./log','testing size')
tf.app.flags.DEFINE_string('base_images_path','../AVA_dataset/AVA_dataset/AVA.txt','testing size')
tf.app.flags.DEFINE_string('ava_dataset_path','../AVA_dataset/AVA_dataset/image/images/','testing size')


sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
if FLAGS.precision_mode == "force_fp32":
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

base_model = MobileNet((FLAGS.image_size, FLAGS.image_size, 3), alpha=1, include_top=False, pooling='avg', weights=None)
base_model.load_weights(os.path.join(FLAGS.TMP_WEIGHTS_PATH+"/mobilenet_1_0_224_tf_no_top.h5"), by_name=True)
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)
model.summary()


# optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam(lr=1e-4)
model.compile(optimizer, loss=earth_mover_loss)


checkpointer = ModelCheckpoint(os.path.join(FLAGS.TMP_MODEL_PATF, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
tensorboard = TensorBoard(log_dir=FLAGS.TMP_LOG_PATH+'/', write_graph=True, update_freq='batch')
callbacks = [checkpointer, tensorboard]


train_generators = train_generator(batchsize=FLAGS.batchsize)
val_generators = val_generator(batchsize=FLAGS.batchsize)


model.fit_generator(train_generators,
                    steps_per_epoch=(FLAGS.train_size // FLAGS.batchsize),
                    epochs=FLAGS.epochs,  callbacks=callbacks,
                    max_queue_size=2,)

sess.close()

