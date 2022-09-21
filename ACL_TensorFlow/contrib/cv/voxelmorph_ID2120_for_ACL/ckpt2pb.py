import tensorflow as tf
from networks import cvpr2018_net

def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''

    tf.reset_default_graph()
    # 定义网络的输入节点
    src = tf.placeholder(dtype=tf.float32, shape=[None, 160, 192, 224, 1], name="input_src")
    tgt = tf.placeholder(dtype=tf.float32, shape=[None, 160, 192, 224, 1], name="input_tgt")
    vol_size = tgt.shape[1:-1]
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # 调用网络模型生成推理图和输出
    y, flow = cvpr2018_net(vol_size, nf_enc, nf_dec, src, tgt)

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "spatial_transformer/map/TensorArrayStack/TensorArrayGatherV3,flow/BiasAdd"

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        print('normal')
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点


# 输入ckpt模型路径
input_checkpoint='./models/model'
# 输出pb模型的路径
out_pb_path="./models/frozen_model.pb"
# 调用freeze_graph将ckpt转为pb
freeze_graph(input_checkpoint,out_pb_path)