import tensorflow as tf
from tensorflow.python.framework import graph_util
import mnasnet_models

ckpt_path = "./models/model.ckpt-10009"

def main():

    tf.reset_default_graph()

    inputs1 = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name="input1")

    with tf.Session() as sess:
        params = {'dropout_rate': 0.2, 'data_format': 'channels_last', 'num_classes': 1000, 'use_keras': True}
        #params = {}
        logits, _ = mnasnet_models.build_mnasnet_model(inputs1, "mnasnet-a1", training=True, override_params=params)
        probs = tf.nn.softmax(logits)
        probs = tf.squeeze(probs, name='pred_prob')

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path) #恢复图并得到数据

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['pred_prob'])  # 如果有多个输出节点，以逗号隔开
        # 保存模型
        with tf.gfile.GFile("./pb_model/mnasnet.pb", "wb") as f: 
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    print("done")

if __name__ == '__main__':
    main()
