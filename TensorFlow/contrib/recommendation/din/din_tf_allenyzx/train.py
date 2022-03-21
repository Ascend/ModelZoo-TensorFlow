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
# Copyright 2020 Huawei Technologies Co., Ltd
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
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from din_model import DIN
from utils import create_amazon_electronic_dataset, DatasetIterater
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.train import AdamOptimizer

tf.app.flags.DEFINE_string('file', './raw_data/remap.pkl', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoint/', '')
tf.app.flags.DEFINE_integer('embed_dim', 8, '')
tf.app.flags.DEFINE_integer('maxlen', 40, '')
tf.app.flags.DEFINE_string('att_hidden_units', '80,40', '')
tf.app.flags.DEFINE_string('ffn_hidden_units', '80,40', '')
tf.app.flags.DEFINE_string('att_activation', 'sigmoid', '')
tf.app.flags.DEFINE_string('ffn_activation', 'prelu', '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_integer('batch_size', 1024, '')
tf.app.flags.DEFINE_float('dnn_dropout', 0.2, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')


FLAGS = tf.app.flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


def main(argv=None):
    # ========================= Hyper Parameters =======================
    file = FLAGS.file
    maxlen = FLAGS.maxlen
    embed_dim = FLAGS.embed_dim
    att_hidden_units = [int(i) for i in FLAGS.att_hidden_units.split(',')]
    ffn_hidden_units = [int(i) for i in FLAGS.ffn_hidden_units.split(',')]
    dnn_dropout = FLAGS.dnn_dropout
    att_activation = FLAGS.att_activation
    ffn_activation = FLAGS.ffn_activation
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    max_steps = FLAGS.max_steps

    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val = create_amazon_electronic_dataset(file, embed_dim, maxlen, cache=False)
    behavior_num = len(behavior_list)
    train_X, train_y = train
    val_X, val_y = val

    #  create input_tensor
    dense_inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='dense_inputs')
    sparse_inputs = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='sparse_inputs')
    seq_inputs = tf.placeholder(shape=(None, maxlen, behavior_num + 1), dtype=tf.int32, name='seq_inputs')
    item_inputs = tf.placeholder(shape=(None, behavior_num + 1), dtype=tf.int32, name='item_inputs')
    y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y_')

    # create model
    din = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
              ffn_activation, maxlen, dnn_dropout, device='gpu')([dense_inputs, sparse_inputs, seq_inputs, item_inputs])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # create loss
    loss = tf.reduce_mean(binary_crossentropy(y_pred=din, y_true=y_))

    learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_rate=0.9, staircase=True,
                                               decay_steps=20000)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # grad
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    train_op = opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=global_step)

    loss_list = []

    def eval(sess, val_X, val_y, model):
        x_dense_inputs, x_sparse_inputs, x_seq_inputs, x_item_inputs = val_X
        x_dense_inputs = np.reshape(x_dense_inputs, (len(x_dense_inputs), 1))
        x_sparse_inputs = np.reshape(x_sparse_inputs, (len(x_sparse_inputs), 1))
        outputs = sess.run([model], feed_dict={dense_inputs: x_dense_inputs,
                                               sparse_inputs: x_sparse_inputs,
                                               seq_inputs: x_seq_inputs,
                                               item_inputs: x_item_inputs})
        auc = roc_auc_score(val_y.reshape(len(val_y), 1), outputs[0])
        acc = accuracy_score(val_y.reshape(len(val_y), 1), np.round(outputs[0]))
        return auc, acc

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        dataset = DatasetIterater(train, batch_size=batch_size, batches_len=len(train_y))
        for step in range(max_steps + 1):
            item = next(dataset)
            x_dense_inputs, x_sparse_inputs, x_seq_inputs, x_item_inputs, train_y = item
            x_dense_inputs = np.reshape(x_dense_inputs, (len(x_dense_inputs), 1))
            x_sparse_inputs = np.reshape(x_sparse_inputs, (len(x_sparse_inputs), 1))
            train_y = np.reshape(train_y, newshape=(len(train_y), 1))
            _, ploss = sess.run([train_op, loss], feed_dict={dense_inputs: x_dense_inputs,
                                                             sparse_inputs: x_sparse_inputs,
                                                             seq_inputs: x_seq_inputs,
                                                             item_inputs: x_item_inputs,
                                                             y_: train_y})
            loss_list.append(ploss)

            if step != 0 and step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)
                print('save: checkpoint to step {}'.format(step))

            if step != 0 and step % 1000 == 0:
                auc, acc = eval(sess=sess, val_X=val_X, val_y=val_y, model=din)
                print("[step:{}] model_loss {:.5} , val_auc {:.5}, acc {:.5}".format(step, float(np.mean(loss_list)),
                                                                                     auc, acc))
            else:
                if step != 0 and step % 200 == 0:
                    print("[step:{}] model_loss {:.5}".format(step, float(np.mean(loss_list))))

        auc, acc = eval(sess=sess, val_X=val_X, val_y=val_y, model=din)
        print("[step:{}] total_loss {:.5} , val_auc {:.5}, acc {:.5}".format(step, float(np.mean(loss_list)),
                                                                             auc, acc))


if __name__ == '__main__':
    tf.app.run()
