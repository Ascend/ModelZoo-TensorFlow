import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import importlib
# network是你自己定义的模型

# 模型的checkpoint文件地址
ckpt_path = "./ckpt_model/model-20190403-164504.ckpt-205000"


def main():
    tf.reset_default_graph()
    network = importlib.import_module("models.resnet34")

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')

    # flow是模型的输出
    prelogits, _ = network.inference(x, 1.0,
                                     phase_train=True, bottleneck_layer_size=512,
                                     weight_decay=0.0)
    flow = tf.nn.l2_normalize(prelogits, 1, 1e-10)
    # flow = network(x)
    # 设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用
    flow = tf.cast(flow, tf.float32, 'out')

    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        # 把图和参数结构一起
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='out',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/frozen_model.pb',
            clear_devices=False,
            initializer_nodes=''
        )

    print("done")


if __name__ == '__main__':
    main()