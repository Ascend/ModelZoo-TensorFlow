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
"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import npu_convert_dropout
import npu_device
import argparse
import ast
#===============================NPU Migration=========================================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                    help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                    help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",
                    help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
parser.add_argument('--data_dir', default='./',help="""directory to data""")
parser.add_argument('--batch_size', default=32, type=int,help="""batch size for 1p""")
parser.add_argument('--epochs', default=2, type=int,help="""epochs""")
parser.add_argument('--static', default=1, type=int,help="""static shape""")
args = parser.parse_args()

def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
      npu_device.global_options().dump_config.enable_dump_debug = True
      npu_device.global_options().dump_config.dump_path = args.over_dump_path
      npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
      npu_device.global_options().profiling_config.enable_profiling = True
      profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                          "training_trace":"on", \
                          "task_trace":"on", \
                          "aicpu":"on", \
                          "aic_metrics":"PipeUtilization",\
                          "fp_point":"", \
                          "bp_point":""}'
      npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode = args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist="../configs/"+args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file="../configs/"+args.fusion_off_file
  npu_device.open().as_default()
#===============================NPU Migration=========================================
npu_config()
data_dir = args.data_dir

"""
## Implement a Transformer block as a layer
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class LossHistory(tf.keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        print('step:%d ,loss: %f ,time:%f'%(batch, loss, time.time() - self.start), flush=True)
    def on_epoch_begin(self, epoch, logs={}):
        self.epochstart = time.time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time.time() - self.epochstart
        print('epoch_duration: ', duration)
        if epoch != 0:
            self.perf.append(duration)
    def on_train_begin(self, logs={}):
        self.batch_size = 24992 // self.params['steps']
        self.samples = self.batch_size * self.params['steps']
        print('params: ', self.params)
        self.perf = []
    def on_train_end(self, logs={}):
        print('imgs/s: %.2f'%(self.samples / np.mean(self.perf)))

logger = LossHistory()

"""
## Download and prepare dataset
"""

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
#(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              static=1,
              **kwargs):
    # Legacy support
    if 'nb_words' in kwargs:
      logging.warning('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
      num_words = kwargs.pop('nb_words')
    if kwargs:
      raise TypeError(f'Unrecognized keyword arguments: {str(kwargs)}.')

    # origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    # path = get_file(
        # path,
        # origin=origin_folder + 'imdb.npz',
        # fi  le_hash=
        # '69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f')
    with np.load(path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
      x_train, labels_train = f['x_train'], f['y_train']
      x_test, labels_test = f['x_test'], f['y_test']

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    if start_char is not None:
      x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
      x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    elif index_from:
      x_train = [[w + index_from for w in x] for x in x_train]
      x_test = [[w + index_from for w in x] for x in x_test]

    if maxlen:
      x_train, labels_train = _remove_long_seq(maxlen, x_train, labels_train)
      x_test, labels_test = _remove_long_seq(maxlen, x_test, labels_test)
      if not x_train or not x_test:
        raise ValueError('After filtering for sequences shorter than maxlen='
                         f'{str(maxlen)}, no sequence was kept. Increase maxlen.')

    xs = x_train + x_test
    labels = np.concatenate([labels_train, labels_test])

    if not num_words:
      num_words = max(max(x) for x in xs)

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
      xs = [
          [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
      ]
    else:
      xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    if static:
        x_train, y_train = np.array(xs[:24992], dtype='object'), labels[:24992]
        x_test, y_test = np.array(xs[idx:idx+24992], dtype='object'), labels[idx:idx+24992]
    else:
        x_train, y_train = np.array(xs[:idx], dtype='object'), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype='object'), labels[idx:]
    return (x_train, y_train), (x_test, y_test)

data_path = os.path.join(data_dir, 'imdb.npz')
(x_train, y_train), (x_val, y_val) = load_data(path=data_path, num_words=vocab_size, static=args.static)

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

"""
## Create classifier model using transformer layer

Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


"""
## Train and Evaluate
"""

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_val, y_val), verbose=2, callbacks=[logger]
)
