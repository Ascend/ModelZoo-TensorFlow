#
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
import ast
import npu_device
import argparse
import tensorflow as tf

from helpers import load_config
from helpers.logging import print_status_bar
from helpers.evaluation import compute_bleu

from sklearn.model_selection import train_test_split
import string
import numpy as np
import io
import os
import time


def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=2, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./data", help="data_path")
    parser.add_argument('--log_steps', type=int, default=100, help='log steps')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
    parser.add_argument('--static', action='store_true', default=False, help='static input shape, default is False')
    parser.add_argument('--learning_rate', type=int, default=1e-4, help='Learning rate for training')
    # ===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str, help='precision mode')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                        help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,
                        help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str, help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str, help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str, help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                        help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                        help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,
                        help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,
                        help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')

    args = parser.parse_args()
    return args


def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode == 'allow_mix_precision':
        npu_device.global_options().modify_mixlist = FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file = FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu_device.global_options().auto_tune_mode = "RL,GA"
    npu_device.open().as_default()


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        self.start_time = None

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size
            print(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d' % (elapsed_time, examples_per_second, self.last_log_step,
                            self.global_steps), flush=True)
            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0


config = load_config("config.json")

dataset_params = config["dataset"]


def clean_sentence(sentence):
    # make a space between each punctionation
    sentence = sentence.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))

    sentence = sentence.strip()  # remove spaces
    return sentence


def preprocess_a_sentence(sentence):
    # clean it
    sentence = clean_sentence(sentence)
    # add the start and end of sequences
    return '<sos> {} <eos>'.format(sentence)


def load_dataset(path, num_examples=None):
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    # list containing a set of (input, output)
    sentence_pairs = [[preprocess_a_sentence(sen) for sen in line.split('\t')] for line in lines[:num_examples]]
    return zip(*sentence_pairs)


def create_shifted_target(y):
    "Remove the start token and append a padding to the end."
    return y[:, :-1], y[:, 1:]


def get_tokenizer(lang, top_k=None):
    # we are keeping the punctionation
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, filters='’,?!"#$%&()*+-/:;=.@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lang)

    sequences = tokenizer.texts_to_sequences(lang)
    # pad the tensors
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
    return sequences, tokenizer


def create_dataset(X, y, batch_size=None, buffer=False, prefetch=tf.data.experimental.AUTOTUNE):
    X_set = tf.data.Dataset.from_tensor_slices(X)
    y_set = tf.data.Dataset.from_tensor_slices(y[0], )
    a_set = tf.data.Dataset.zip((X_set, y_set))
    if buffer:
        a_set = a_set.shuffle(X[0].shape[0])
    if batch_size is not None:
        a_set = a_set.batch(batch_size, drop_remainder=True)
    return a_set.repeat().prefetch(prefetch)


def padded_transform(X, tokenizer, X_max):
    X = tokenizer.texts_to_sequences(X)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post", maxlen=X_max)
    return X


def dataset_padded_transform(X, y, X_tokenizer, y_tokenizer, X_max, y_max):
    X = padded_transform(X, X_tokenizer, X_max)
    y = padded_transform(y, y_tokenizer, y_max)
    return X, y


def dataset(input_lang, target_lang, batch_size, prefetch=tf.data.experimental.AUTOTUNE,
            valid_size=0.15, top_k=None):
    encoder_train, encoder_valid, target_train, target_valid = train_test_split(input_lang,
                                                                                target_lang,
                                                                                test_size=valid_size)
    # build tokenizer
    encoder_train, input_tokenizer = get_tokenizer(encoder_train, top_k=top_k)
    target_train, target_tokenizer = get_tokenizer(target_train, top_k=top_k)

    # transform and pad
    encoder_valid, target_valid = dataset_padded_transform(encoder_valid, target_valid,
                                                           input_tokenizer, target_tokenizer,
                                                           encoder_train.shape[1], target_train.shape[1])

    decoder_train, y_train = create_shifted_target(target_train)
    train_attention_weights = np.zeros((len(decoder_train)), dtype=np.float32)

    decoder_valid, y_valid = create_shifted_target(target_valid)
    valid_attention_weights = np.zeros((len(decoder_valid)), dtype=np.float32)

    # create dataset    
    train_set = create_dataset((encoder_train, decoder_train),
                               (y_train, train_attention_weights),
                               batch_size=batch_size, buffer=True,
                               prefetch=prefetch)

    valid_set = create_dataset((encoder_valid, decoder_valid),
                               (y_valid, valid_attention_weights),
                               batch_size=batch_size, prefetch=prefetch)

    # information about the training set:
    info = dict(
        train_size=encoder_train.shape[0],
        train_input_max_pad=encoder_train.shape[1],
        train_target_max_pad=target_train.shape[1],
        valid_size=encoder_valid.shape[0],
    )
    return train_set, valid_set, info, input_tokenizer, target_tokenizer


def main():
    args = get_args()
    npu_config(args)
    DATA_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    TRAIN_EPOCHS = args.train_epochs
    MODEL_DIR = args.model_dir
    LOG_STEPS = args.log_steps
    LEARNING_RATE = args.learning_rate

    # Load configuration
    batch_size = BATCH_SIZE
    num_examples = dataset_params["num_examples"]
    top_k = dataset_params['top_common_words']

    # load dataset and split training, validation and testing sets.
    path_to_file = os.path.join(DATA_PATH, 'fra.txt')
    target_lang, input_lang, _ = load_dataset(path_to_file, num_examples=num_examples)
    encoder_train, encoder_test, target_train, target_test = train_test_split(input_lang,
                                                                              target_lang,
                                                                              test_size=0.2)
    # create training and validation set
    train_set, valid_set, info, input_tokenizer, target_tokenizer = dataset(encoder_train, target_train, batch_size,
                                                                            top_k=top_k)

    for x, y, in train_set.take(1):
        i = 10
        enc_x, dec_x = x
        # y, att = y
        print(input_tokenizer.sequences_to_texts([enc_x[i].numpy()]))
        print(target_tokenizer.sequences_to_texts([dec_x[i].numpy()]))
        print(target_tokenizer.sequences_to_texts([y[i].numpy()]))
        # print(att.shape)

    model_config = config['model']
    N = model_config['N']
    model_depth = model_config['model_depth']
    num_heads = model_config['num_heads']
    dff = model_config['dff']
    dropout_rate = model_config['dropout_rate']
    epochs = model_config['epochs']

    steps_per_epoch = info['train_size'] // batch_size
    validation_steps = info['valid_size'] // batch_size
    max_input_vocab = len(input_tokenizer.index_word) + 1
    max_target_vocab = len(target_tokenizer.index_word) + 1
    input_max_positional_encoding = max_input_vocab
    target_max_positional_encoding = max_target_vocab

    from model import TransformerV2, CustomSchedule

    transformer = TransformerV2(N, model_depth, num_heads, dff,
                                max_input_vocab, max_target_vocab,
                                input_max_positional_encoding, target_max_positional_encoding,
                                rate=dropout_rate)

    adam = tf.keras.optimizers.Adam(CustomSchedule(model_depth), beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)
    transformer.compile(optimizer=adam, metrics=['accuracy'], loss='sparse_categorical_crossentropy')

    history = transformer.fit(train_set, steps_per_epoch=steps_per_epoch, epochs=TRAIN_EPOCHS,
                              validation_data=valid_set, validation_steps=validation_steps,
                              verbose=2, callbacks=[TimeHistory(BATCH_SIZE, LOG_STEPS)])

    transformer.save_weights(filepath=os.path.join(MODEL_DIR, 'tf_model'), save_format='tf')


if __name__ == '__main__':
    main()
