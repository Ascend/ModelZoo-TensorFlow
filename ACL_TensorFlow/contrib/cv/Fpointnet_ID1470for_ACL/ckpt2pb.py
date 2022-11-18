import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

tf.compat.v1.disable_eager_execution()

def freeze_graph(input_checkpoint,output_graph):
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "Squeeze_1,add_2"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        
        for node in graph_def.node:   
            if node.op == 'Placeholder' and node.name == 'Placeholder_8':
                print(node)
                node.op = 'Const'
                node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(False, tf.bool)))
                del node.attr['shape']
                
            if node.op == 'PyFunc' and node.name == 'PyFunc':
                print(node)
                node.op = 'Placeholder'
                node.name = 'Placeholder_9'
                shape = tf.TensorShape([32,None,2])
                node.attr["shape"].CopyFrom(tf.AttrValue(shape=shape.as_proto()))
                dtype = tf.dtypes.as_dtype("int32")
                node.attr["dtype"].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
                node.input.remove('Squeeze_1')
                del node.attr['Tin']
                del node.attr['Tout']
                del node.attr['token']
                del node.attr['_output_shapes']
                print(node)        
                
            if node.op == 'GatherNd' and node.name == 'GatherNd':
                node.input.remove('PyFunc')
                node.input.append('Placeholder_9')
                print(node)
                
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

input_checkpoint='./model.ckpt'
out_pb_path='fpointnet.pb'
freeze_graph(input_checkpoint, out_pb_path)
