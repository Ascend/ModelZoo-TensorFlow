import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

ckpt_path = 'pre_trained/ag_news/model.ckpt'
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main():
    tf.reset_default_graph()

    mx_len = 126
    inp = tf.placeholder(tf.int32, shape=(None, mx_len), name='input')
    weights = tf.placeholder(tf.float32, shape=(None, mx_len), name='weights')
    labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
    learning_rate = 0.3
    dropout_rate = 0.5
    is_training = False
    num_words_in_train = 188111
    embedding_dim = 10
    use_batch_norm = False
    num_labels = 4

    with tf.name_scope('embeddings'):
        token_embeddings = tf.Variable(tf.random.uniform([num_words_in_train, embedding_dim]), name='embedding_matrix')

    with tf.name_scope('mean_sentence_embedding'):
        gathered_embeddings = tf.gather(token_embeddings, inp)
        weights_broadcasted = tf.expand_dims(weights, axis=2)
        mean_embedding = tf.reduce_sum(tf.multiply(weights_broadcasted, gathered_embeddings), axis=1, name='sentence_embedding')

    if use_batch_norm:
        mean_embedding = tf.layers.batch_normalization(mean_embedding, training=is_training)
    mean_embedding_dropout = tf.layers.dropout(mean_embedding, rate=dropout_rate, training=is_training)
    
    logits = tf.layers.dense(mean_embedding_dropout, num_labels, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(), name='logits')

    predict_class = tf.nn.softmax(logits, name='prediction')
   
    with tf.Session(config=config) as sess:
        tf.train.write_graph(sess.graph_def, 'save', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='save/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='prediction',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='save/fasttext.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

if __name__ == '__main__':
    main()
