from tensorflow.python.framework import graph_util
import tensorflow as tf
def freeze_graph(input_checkpoint,output_graph):
  '''
  :param input_checkpoint:
  :param output_graph: PB模型保存路径
  :return:
  '''


  # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
  output_node_names = "out_afm,interactions,train_features_afm,dropout_keep_afm,train_phase_afm,train_labels_afm"
  #output_node_names = "train_features_afm,train_labels_afm,dropout_keep_afm,train_phase_afm"
  saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
  graph = tf.get_default_graph() # 获得默认的图
  input_graph_def = graph.as_graph_def() # 返回一个序列化的图代表当前的图

  with tf.Session() as sess:
    saver.restore(sess, input_checkpoint) #恢复图并得到数据
    output_graph_def = graph_util.convert_variables_to_constants( # 模型持久化，将变量值固定
    sess=sess,
    input_graph_def=input_graph_def,# 等于:sess.graph_def
    output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
    with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
      f.write(output_graph_def.SerializeToString()) #序列化输出
      print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
    
    # for op in graph.get_operations():
    # print(op.name, op.values())

input_checkpoint="../pretrain/afm_ml-tag_256/ml-tag_256"
output_graph = "../om/model/afm.pb"
freeze_graph(input_checkpoint,output_graph)