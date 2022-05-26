import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# 导入网络模型文件
from tensorflow.contrib import slim as slim
import numpy as np
from core.PwcNet.pwcnet import pwcnet

# 指导ckpt路径
ckpt_path = "./ckpt5/model-99999"

def main():
    tf.reset_default_graph()
    # 定义输入节点
    inputs = tf.placeholder(dtype=tf.float32, shape=(1, 2, 384, 1280, 3), name="input")
    # 调用网络模型生成推理图
    pred_flows, pred_pyrs = pwcnet(inputs)

    # 定义输出节点
    predict = tf.identity(pred_flows, name='output')
    with tf.Session() as sess:
        tf.io.write_graph(sess.graph_def, './pb_model_tf', 'flow_model.pb')    # 默认，不需要修改
        freeze_graph.freeze_graph(
              input_graph='./pb_model_tf/flow_model.pb',   # 默认，不需要修改
              input_saver='',
              input_binary=False,
              input_checkpoint=ckpt_path,
              output_node_names='output',  # 与上面定义的输出节点一致
              restore_op_name='save/restore_all',
              filename_tensor_name='save/Const:0',
              output_graph='./pb_model_tf/flow.pb',   # 改为对应网络的名称
              clear_devices=False,
              initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()