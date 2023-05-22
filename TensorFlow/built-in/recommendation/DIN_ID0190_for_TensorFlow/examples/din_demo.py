from npu_bridge.npu_init import *
import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepctr.models import DIN
import numpy as np
import tensorflow.python.keras as keras
rank_size = int(os.getenv('RANK_SIZE',"1"))
rank_id = int(os.getenv('RANK_ID'))

def split_tfrecord(tfrecord_path):
    if rank_size == 1:
        return tfrecord_path
    else:
        if not os.path.exists("./data/" + f'train_{rank_size}_part{rank_id}_.tfrecord.gz'):
            with tf.Graph().as_default(), tf.Session() as sess:
                # The "batch" here does not effect the training batch actually. It's just for speeding up the process.
                # Another option without "batch" is:
                ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP").shard(rank_size, rank_id).batch(1024)
                batch = ds.make_one_shot_iterator().get_next()
                with tf.python_io.TFRecordWriter("./data/" + f'train_{rank_size}_part{rank_id}.tfrecord.gz', tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
                    while True:
                        try:
                            records = sess.run(batch)
                            for record in records:
                                writer.write(record)
                        except tf.errors.OutOfRangeError: break

        return "./data/" + f'train_{rank_size}_part{rank_id}.tfrecord.gz'

def input_fn(filenames, is_train, batch_size=1024):
    def _parse_function(example_proto):
        feature_description = {
            "movieId": tf.io.FixedLenFeature([1], tf.int64),
            "cateId": tf.io.FixedLenFeature([1], tf.int64),
            "hist_movieId": tf.io.FixedLenFeature([200], tf.int64),
            "hist_cateId": tf.io.FixedLenFeature([200], tf.int64),
            "seq_length": tf.io.FixedLenFeature([1], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32)
        }
        features = tf.io.parse_example(example_proto, feature_description)
        labels = features.pop("label")
        return features, labels

    dataset = tf.data.TFRecordDataset(filenames=filenames, compression_type="GZIP")
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(drop_remainder=True, batch_size=batch_size)
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=24)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    config_proto = tf.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.ON
    custom_op.parameter_map["hcom_parallel"].b = True
    #custom_op.parameter_map["enable_data_pre_proc"].b = True
    npu_keras_sess = set_keras_session_npu_config(config=config_proto)

    # shard for 8p
    filename = split_tfrecord(r"./data/train.tfrecords.gz")
    # set din features
    feature_columns = [SparseFeat(name='movieId', vocabulary_size=131263, embedding_dim=64),
                       SparseFeat(name='cateId', vocabulary_size=21, embedding_dim=64),
                       VarLenSparseFeat(SparseFeat('hist_movieId', vocabulary_size=131263, embedding_dim=64,
                                                   embedding_name='movieId'), maxlen=200, length_name="seq_length"),
                       VarLenSparseFeat(SparseFeat('hist_cateId', vocabulary_size=21, embedding_dim=64,
                                                   embedding_name='cateId'), maxlen=200, length_name="seq_length")
                       ]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["movieId", "cateId"]
    model = DIN(feature_columns, behavior_feature_list)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3 * rank_size)
    opt = npu_distributed_optimizer_wrapper(opt)
    model.compile(opt, 'binary_crossentropy', metrics=['binary_crossentropy', "AUC"])

    callbacks = [NPUBroadcastGlobalVariablesCallback(0)]

    model.fit(x=input_fn(filename, True), epochs=5, verbose=1,
              validation_data=input_fn(r"./data/test.tfrecords.gz", False), validation_steps=5406, callbacks=callbacks)

