from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicGRUV2
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicAUGRU
import tensorflow as tf

# from tensorflow.python.ops.rnn_cell import LSTMCell
# from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from my_rnn import dynamic_rnn
# from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import GRUCell
from utils import *
from Dice import dice

class Model(object):
    def __init__(self, batch_size, maxlen, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [batch_size, maxlen], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [batch_size, maxlen], name='cat_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [batch_size, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [batch_size, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [batch_size, ], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [batch_size, maxlen], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [batch_size], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [batch_size, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling =use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [batch_size, maxlen, 5], name='noclk_mid_batch_ph') #generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [batch_size, maxlen, 5], name='noclk_cat_batch_ph')
            
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = embedding_lookup_npu(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = embedding_lookup_npu(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = embedding_lookup_npu(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = embedding_lookup_npu(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = embedding_lookup_npu(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = embedding_lookup_npu(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = embedding_lookup_npu(self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)# 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], EMBEDDING_DIM * 2])# cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        print(">>>>>>>>>>>>>>>>>>>>bn1", bn1.get_shape().as_list())
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-04).minimize(self.loss)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # grads = self.optimizer.compute_gradients(self.loss * (2**12))
            # grads = [(grad / (2**12), var) for grad, var in grads]
            # grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
            # self.optimizer = self.optimizer.apply_gradients(grads)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_sum(click_loss_ + noclick_loss_) / (tf.reduce_sum(mask) + 1e-6)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        print(">>>>>>>>>>>>>>>>>>>>>in_", in_.get_shape().as_list())
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            # outputs = sess.run(self.rnn_outputs, feed_dict={
            #     self.uid_batch_ph: inps[0],
            #     self.mid_batch_ph: inps[1],
            #     self.cat_batch_ph: inps[2],
            #     self.mid_his_batch_ph: inps[3],
            #     self.cat_his_batch_ph: inps[4],
            #     self.mask: inps[5],
            #     self.target_ph: inps[6],
            #     self.seq_len_ph: inps[7],
            #     self.lr: inps[8],
            #     self.noclk_mid_batch_ph: inps[9],
            #     self.noclk_cat_batch_ph: inps[10],
            # })
            # print(outputs)
            # return

            # op = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # for key in op:
            #     if "dynamic_gru_v2" in key.name:
            #         print(key.name, sess.run(key))

            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                self.noclk_mid_batch_ph: inps[9],
                self.noclk_cat_batch_ph: inps[10],
            })
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>mid_his_batch_ph")
            # print(inps[3])
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>embedding")
            # print(item_his_eb_time_major)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>output")
            # print(outputs)
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _, summary_str = sess.run([self.loss, self.accuracy, self.optimizer, self.merged], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
            })
            return loss, accuracy, 0, summary_str

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.noclk_mid_batch_ph: inps[8],
                self.noclk_cat_batch_ph: inps[9],
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
    
    def summary_op(self, summary_writer, summary_str, step):
        summary_writer.add_summary(summary_str, global_step=step)

# DIEN
class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self, batch_size, maxlen, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(batch_size, maxlen, n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            # rnn_outputs, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
            #                              sequence_length=self.seq_len_ph, dtype=tf.float32,
            #                              scope="gru1")

            item_his_eb_fp16 = tf.cast(self.item_his_eb, tf.float16, name="cast_fp16")
            item_his_eb_time_major = tf.transpose(item_his_eb_fp16, [1, 0, 2], name="transpose_time_major")
            gruv2 = DynamicGRUV2(HIDDEN_SIZE, dtype=tf.float16)
            rnn_outputs, _, _, _, _, _ = gruv2(item_his_eb_time_major)
            rnn_outputs_time_major = tf.transpose(rnn_outputs, [1, 0, 2], name="rnn_outputs_transpose_time_major")
            rnn_outputs = tf.cast(rnn_outputs_time_major, tf.float32)
            # mask_dim = tf.expand_dims(self.mask, -1)
            # rnn_outputs = rnn_outputs * mask_dim
            # print("?????????????????rnn_outputs", rnn_outputs)

            # item_his_eb_fp16 = tf.cast(self.item_his_eb, tf.float16, name="cast_fp16")
            # gru = tf.keras.layers.CuDNNGRU(HIDDEN_SIZE, return_sequences=True)
            # rnn_outputs = gru(item_his_eb_fp16)
            # # print(rnn_outputs)
            # self.rnn_outputs = rnn_outputs
            # rnn_outputs = tf.cast(rnn_outputs, tf.float32)

            # from tensorflow.contrib.cudnn_rnn.python.layers import CudnnGRU
            # from tensorflow.python.framework import dtypes

            # item_his_eb_fp16 = tf.cast(self.item_his_eb, tf.float16, name="cast_fp16")
            # gru = CudnnGRU(num_layers=1, num_units=HIDDEN_SIZE, dtype=dtypes.float16)
            # rnn_outputs, _ = gru(inputs=item_his_eb_fp16, sequence_lengths=self.seq_len_ph)
            # # print(rnn_outputs)
            # self.rnn_outputs = rnn_outputs
            # rnn_outputs = tf.cast(rnn_outputs, tf.float32)

            # rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
            #                              sequence_length=self.seq_len_ph, dtype=tf.float32,
            #                              scope="gru1")
            # self.rnn_outputs = rnn_outputs

            # item_his_eb_fp16 = tf.cast(self.item_his_eb, tf.float16, name="cast_fp16")
            # rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=item_his_eb_fp16,
            #                              sequence_length=self.seq_len_ph, dtype=tf.float16,
            #                              scope="gru1")
            # rnn_outputs = tf.cast(rnn_outputs, tf.float32)
            # self.rnn_outputs = rnn_outputs
            
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs_fp16 = tf.cast(rnn_outputs, tf.float16)
            alphas_fp16 = tf.cast(alphas, tf.float16)
            rnn_outputs_time_major = tf.transpose(rnn_outputs_fp16, [1, 0, 2], name="gru2_transpose_time_major")
            alphas_fp16_time_major = tf.transpose(alphas_fp16, [1, 0], name="att_transpose_time_major")

            augru = DynamicAUGRU(HIDDEN_SIZE, dtype=tf.float16)
            rnn_outputs2, _, _, _, _, _, _ = augru(rnn_outputs_time_major, alphas_fp16_time_major)

            rnn_outputs2_time_major = tf.transpose(rnn_outputs2, [1, 0, 2], name="gru2_rnn_outputs_transpose_time_major")
            rnn_outputs2 = tf.cast(rnn_outputs2_time_major, tf.float32)
            final_state2 = tf.batch_gather(rnn_outputs2, self.seq_len_ph[:, None] - 1)
            final_state2 = tf.squeeze(final_state2, 1)

            # rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
            #                                          att_scores = tf.expand_dims(alphas, -1),
            #                                          sequence_length=self.seq_len_ph, dtype=tf.float32,
            #                                          scope="gru2")
            # rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
            #                                          att_scores = tf.expand_dims(alphas, -1),
            #                                          dtype=tf.float32,
            #                                          scope="gru2")
            # final_state2 = tf.batch_gather(rnn_outputs2, self.seq_len_ph[:, None]-1)
            # final_state2 = tf.squeeze(final_state2, 1)
            

            # rnn_outputs2, final_state2 = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
            #                                                 sequence_length=self.seq_len_ph, dtype=tf.float32,
            #                                                 scope="gru2")
            # print("@@@@@@@@@@@@@@@@@@@", rnn_outputs2.get_shape().as_list())
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN_V2_Gru_Vec_attGru, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.uid_batch_embedded, self.item_eb, final_state2, self.item_his_eb_sum], 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


@tf.custom_gradient
def gather_npu(params, indices):
  def grad(dy):
    params_shape = tf.shape(params, out_type=tf.int64)
    params_shape = tf.cast(params_shape, tf.int32)
    grad_gather = tf.unsorted_segment_sum(dy, indices, params_shape[0])
    return grad_gather, None
  return tf.gather(params, indices), grad

@tf.custom_gradient
def embedding_lookup_npu(params, indices):
  def grad(dy):
    params_shape = tf.shape(params, out_type=tf.int64)
    params_shape = tf.cast(params_shape, tf.int32)
    grad_embedding_lookup = tf.unsorted_segment_sum(dy, indices, params_shape[0])
    return grad_embedding_lookup, None
  return tf.nn.embedding_lookup(params, indices), grad