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
Title: Text classification with Switch Transformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/05/10
Last modified: 2021/02/15
Description: Implement a Switch Transformer for text classification.
"""

"""
## Introduction

This example demonstrates the implementation of the
[Switch Transformer](https://arxiv.org/abs/2101.03961) model for text
classification.

The Switch Transformer replaces the feedforward network (FFN) layer in the standard
Transformer with a Mixture of Expert (MoE) routing layer, where each expert operates
independently on the tokens in the sequence. This allows increasing the model size without
increasing the computation needed to process each example.

Note that, for training the Switch Transformer efficiently, data and model parallelism
need to be applied, so that expert modules can run simultaneously, each on its own accelerator.
While the implementation described in the paper uses the
[TensorFlow Mesh](https://github.com/tensorflow/mesh) framework for distributed training,
this example presents a simple, non-distributed implementation of the Switch Transformer
model for demonstration purposes.
"""

"""
## Setup
"""

import npu_device
#npu_device.open().as_default()
import ast
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import npu_convert_dropout
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default='./',
                        help="""directory to data""")
    parser.add_argument('--batch_size', default=50, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=3, type=int,help="""epochs""")
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10",help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval,help='auto_tune flag, default is False')
    parser.add_argument('--static', default=False, type=bool,help="""static""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()
data_dir = args.data_dir

"""
## Download and prepare dataset
"""
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
        npu_device.global_options().modify_mixlist=args.mixlist_file
    if args.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=args.fusion_off_file
    if args.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()
#===============================NPU Migration=========================================
npu_config()
vocab_size = 20000  # Only consider the top 20k words
num_tokens_per_example = 200  # Only consider the first 200 words of each movie review

def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              static=False,
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
        #x_train, y_train = np.array(xs[idx-24992:idx], dtype='object'), labels[idx-24992:idx]
        #x_test, y_test = np.array(xs[idx:idx+24992], dtype='object'), labels[idx:idx+24992]
        x_train, y_train = np.array(xs[:idx], dtype='object'), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype='object'), labels[idx:]
    else:
        x_train, y_train = np.array(xs[:idx], dtype='object'), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype='object'), labels[idx:]
    return (x_train, y_train), (x_test, y_test)

data_path = os.path.join(data_dir, 'imdb.npz')
(x_train, y_train), (x_val, y_val) = load_data(path=data_path, num_words=vocab_size,static=args.static)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=num_tokens_per_example
)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=num_tokens_per_example)

"""
## Define hyperparameters
"""

embed_dim = 32  # Embedding size for each token.
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feedforward network.
num_experts = 10  # Number of experts used in the Switch Transformer.
batch_size = args.batch_size  # Batch size.
learning_rate = 0.001  # Learning rate.
dropout_rate = 0.25  # Dropout rate.
num_epochs = args.epochs  # Number of epochs.
num_tokens_per_batch = (
    batch_size * num_tokens_per_example
)  # Total number of tokens per batch.
print(f"Number of tokens per batch: {num_tokens_per_batch}")

"""
## Implement token & position embedding layer

It consists of two seperate embedding layers, one for tokens, one for token index (positions).
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


"""
## Implement the feedforward network

This is used as the Mixture of Experts in the Switch Transformer.
"""


def create_feedforward_network(ff_dim, name=None):
    return keras.Sequential(
        [layers.Dense(ff_dim, activation="relu"), layers.Dense(ff_dim)], name=name
    )


"""
## Implement the load-balanced loss

This is an auxiliary loss to encourage a balanced load across experts.
"""


def load_balanced_loss(router_probs, expert_mask):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = tf.shape(expert_mask)[-1]
    # Get the fraction of tokens routed to each expert.
    # density is a vector of length num experts that sums to 1.
    density = tf.reduce_mean(expert_mask, axis=0)
    # Get fraction of probability mass assigned to each expert from the router
    # across all tokens. density_proxy is a vector of length num experts that sums to 1.
    density_proxy = tf.reduce_mean(router_probs, axis=0)
    # Want both vectors to have uniform allocation (1/num experts) across all
    # num_expert elements. The two vectors will be pushed towards uniform allocation
    # when the dot product is minimized.
    loss = tf.reduce_mean(density_proxy * density) * tf.cast(
        (num_experts ** 2), tf.dtypes.float32
    )
    return loss


"""
### Implement the router as a layer
"""


class Router(layers.Layer):
    def __init__(self, num_experts, expert_capacity):
        self.num_experts = num_experts
        self.route = layers.Dense(units=num_experts)
        self.expert_capacity = expert_capacity
        super(Router, self).__init__()

    def call(self, inputs, training=False):
        # inputs shape: [tokens_per_batch, embed_dim]
        # router_logits shape: [tokens_per_batch, num_experts]
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.
            router_logits += tf.random.uniform(
                shape=router_logits.shape, minval=0.9, maxval=1.1
            )
        # Probabilities for each token of what expert it should be sent to.
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        # Get the top−1 expert for each token. expert_gate is the top−1 probability
        # from the router for each token. expert_index is what expert each token
        # is going to be routed to.
        expert_gate, expert_index = tf.math.top_k(router_probs, k=1)
        # expert_mask shape: [tokens_per_batch, num_experts]
        expert_mask = tf.one_hot(expert_index, depth=self.num_experts)
        # Compute load balancing loss.
        aux_loss = load_balanced_loss(router_probs, expert_mask)
        self.add_loss(aux_loss)
        # Experts have a fixed capacity, ensure we do not exceed it. Construct
        # the batch indices, to each expert, with position in expert make sure that
        # not more that expert capacity examples can be routed to each expert.
        position_in_expert = tf.cast(
            tf.math.cumsum(expert_mask, axis=0) * expert_mask, tf.dtypes.int32
        )
        # Keep only tokens that fit within expert capacity.
        expert_mask *= tf.cast(
            tf.math.less(
                tf.cast(position_in_expert, tf.dtypes.int32), self.expert_capacity
            ),
            tf.dtypes.float32,
        )
        expert_mask_flat = tf.reduce_sum(expert_mask, axis=-1)
        # Mask out the experts that have overflowed the expert capacity.
        expert_gate *= expert_mask_flat
        # Combine expert outputs and scaling with router probability.
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        combined_tensor = tf.expand_dims(
            expert_gate
            * expert_mask_flat
            * tf.squeeze(tf.one_hot(expert_index, depth=self.num_experts), 1),
            -1,
        ) * tf.squeeze(tf.one_hot(position_in_expert, depth=self.expert_capacity), 1)
        # Create binary dispatch_tensor [tokens_per_batch, num_experts, expert_capacity]
        # that is 1 if the token gets routed to the corresponding expert.
        dispatch_tensor = tf.cast(combined_tensor, tf.dtypes.float32)

        return dispatch_tensor, combined_tensor


"""
### Implement a Switch layer
"""


class Switch(layers.Layer):
    def __init__(self, num_experts, embed_dim, num_tokens_per_batch, capacity_factor=1):
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.experts = [
            create_feedforward_network(embed_dim) for _ in range(num_experts)
        ]

        self.expert_capacity = num_tokens_per_batch // self.num_experts
        self.router = Router(self.num_experts, self.expert_capacity)
        super(Switch, self).__init__()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_tokens_per_example = tf.shape(inputs)[1]

        # inputs shape: [num_tokens_per_batch, embed_dim]
        inputs = tf.reshape(inputs, [num_tokens_per_batch, self.embed_dim])
        # dispatch_tensor shape: [expert_capacity, num_experts, tokens_per_batch]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        # expert_inputs shape: [num_experts, expert_capacity, embed_dim]
        expert_inputs = tf.einsum("ab,acd->cdb", inputs, dispatch_tensor)
        expert_inputs = tf.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = tf.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = tf.stack(expert_output_list, axis=1)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = tf.einsum(
            "abc,xba->xc", expert_outputs, combine_tensor
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]
        outputs = tf.reshape(
            expert_outputs_combined,
            [batch_size, num_tokens_per_example, self.embed_dim],
        )
        return outputs


"""
## Implement a Transformer block layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # The ffn can be either a standard feedforward network or a switch
        # layer with a Mixture of Experts.
        self.ffn = ffn
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement the classifier

The `TransformerBlock` layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and use a feedforward network on top
of it to classify text.
"""


def create_classifier():
    switch = Switch(num_experts, embed_dim, num_tokens_per_batch)
    transformer_block = TransformerBlock(ff_dim, num_heads, switch)

    inputs = layers.Input(shape=(num_tokens_per_example,))
    embedding_layer = TokenAndPositionEmbedding(
        num_tokens_per_example, vocab_size, embed_dim
    )
    x = embedding_layer(inputs)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    classifier = keras.Model(inputs=inputs, outputs=outputs)
    return classifier


"""
## Train and evaluate the model
"""


def run_experiment(classifier):
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = classifier.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        verbose=2
    )
    classifier.save_weights(filepath="tf_model", save_format="tf")
    return history


classifier = create_classifier()
run_experiment(classifier)


"""
## Conclusion

Compared to the standard Transformer architecture, the Switch Transformer can have a much
larger number of parameters, leading to increased model
capacity, while maintaining a reasonable computational cost.
"""
