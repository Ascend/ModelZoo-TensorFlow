import os
import tensorflow as tf
from model import Model
from config import cfg
from tensorflow.python.tools import freeze_graph


ckpt_path = "cache/dataset/COCO/snapshot_140.ckpt"
result_dir = cfg.pb_dir

tf.reset_default_graph()
model = Model()
model.make_network(is_train=False)

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, result_dir, 'model.pb')

    freeze_graph.freeze_graph(
        input_graph=os.path.join(result_dir, 'model.pb'),
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_path,
        output_node_names='out/BiasAdd',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(result_dir, 'frozen_model.pb'),
        clear_devices=False,
        initializer_nodes=''
    )
