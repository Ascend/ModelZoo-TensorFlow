import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util


def freeze_graph(ckpt_path, pb_model):
    output_node_names = "probs"
    saver = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(".")
        )

        with tf.gfile.GFile(pb_model, "wb") as f:
            f.write(output_graph_def.SerializeToString())


new_model_path = 'D:/Log/snap-23501_final'  # dataset形式改成placeholder形式
pb_path = 'D:/Log/randlanet_final.pb'  # pb格式的模型

freeze_graph(new_model_path, pb_path)

